import torch
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import WarmupScheduler
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]))


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               phase_1_steps,
               phase_2_steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               lang="en",
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               resume=False,
               warmup_steps=4000,
               use_wandb=False,
               kl_start_steps=10000,
               postnet_start_steps=160000):
    # ============
    # Preparations
    # ============
    steps = phase_1_steps + phase_2_steps
    net = net.to(device)

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=1,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        shuffle=True,
                                        prefetch_factor=5,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    grad_scaler = GradScaler()
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    if resume:
        previous_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if previous_checkpoint is not None:
            path_to_checkpoint = previous_checkpoint
        else:
            raise RuntimeError(f"No checkpoint found that can be resumed from in {save_directory}")
    step_counter = 0
    train_losses_total = list()
    l1_losses_total = list()
    ssim_losses_total = list()
    mse_losses_total = list()
    duration_losses_total = list()
    pitch_losses_total = list()
    energy_losses_total = list()
    kl_losses_total = list()
    glow_losses_total = list()
    cycle_losses_total = list()
    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if resume:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            grad_scaler.load_state_dict(check_dict["scaler"])
            scheduler.load_state_dict(check_dict["scheduler"])
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return

    net.train()
    # =============================
    # Actual train loop starts here
    # =============================
    for step in tqdm(range(step_counter, steps)):
        batches = []
        while len(batches) < batch_size:
            for index in random.sample(list(range(len(datasets))), len(datasets)):
                if len(batches) < batch_size:
                    # we get one batch for each task (i.e. language in this case) in a randomized order
                    try:
                        batch = next(train_iters[index])
                        batches.append(batch)
                    except StopIteration:
                        train_iters[index] = iter(train_loaders[index])
                        batch = next(train_iters[index])
                        batches.append(batch)
        batch = collate_and_pad(batches)
        train_loss = 0.0
        with autocast():
            if step <= phase_1_steps:
                # PHASE 1
                # we sum the loss for each task, as we would do for the
                # second order regular MAML, but we do it only over one
                # step (i.e. iterations of inner loop = 1)

                style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                           batch_of_spectrogram_lengths=batch[3].to(device))

                l1_loss, ssim_loss, mse_loss, duration_loss, pitch_loss, energy_loss, kl_loss, glow_loss = net(text_tensors=batch[0].to(device),
                                                                                                               text_lengths=batch[1].to(device),
                                                                                                               gold_speech=batch[2].to(device),
                                                                                                               speech_lengths=batch[3].to(device),
                                                                                                               gold_durations=batch[4].to(device),
                                                                                                               gold_pitch=batch[6].to(device),
                                                                                                               # mind the switched order
                                                                                                               gold_energy=batch[5].to(device),
                                                                                                               # mind the switched order
                                                                                                               utterance_embedding=style_embedding,
                                                                                                               lang_ids=batch[8].to(device),
                                                                                                               return_mels=False,
                                                                                                               run_glow=step_counter > postnet_start_steps)

                train_loss = train_loss + l1_loss + ssim_loss + mse_loss + duration_loss + pitch_loss + energy_loss
                if step_counter > postnet_start_steps:
                    train_loss = train_loss + glow_loss
                if step_counter > kl_start_steps:
                    train_loss = train_loss + kl_loss

            else:
                # PHASE 2
                # cycle objective is added to make sure the embedding function is given adequate attention
                style_embedding_function.eval()
                style_embedding_of_gold, out_list_gold = style_embedding_function(
                    batch_of_spectrograms=batch[2].to(device),
                    batch_of_spectrogram_lengths=batch[3].to(device),
                    return_all_outs=True)

                l1_loss, ssim_loss, mse_loss, duration_loss, pitch_loss, energy_loss, kl_loss, glow_loss, output_spectrograms = net(
                    text_tensors=batch[0].to(device),
                    text_lengths=batch[1].to(device),
                    gold_speech=batch[2].to(device),
                    speech_lengths=batch[3].to(device),
                    gold_durations=batch[4].to(device),
                    gold_pitch=batch[6].to(device),
                    # mind the switched order
                    gold_energy=batch[5].to(device),
                    # mind the switched order
                    utterance_embedding=style_embedding_of_gold.detach(),
                    lang_ids=batch[8].to(device),
                    return_mels=True,
                    run_glow=step_counter > postnet_start_steps)

                train_loss = train_loss + l1_loss + ssim_loss + mse_loss + duration_loss + pitch_loss + energy_loss
                if step_counter > postnet_start_steps:
                    train_loss = train_loss + glow_loss
                if step_counter > kl_start_steps:
                    train_loss = train_loss + kl_loss

                style_embedding_function.train()
                style_embedding_of_predicted, out_list_predicted = style_embedding_function(
                    batch_of_spectrograms=output_spectrograms,
                    batch_of_spectrogram_lengths=batch[3].to(device),
                    return_all_outs=True)

                cycle_dist = 0
                for out_gold, out_pred in zip(out_list_gold, out_list_predicted):
                    # essentially feature matching, as is often done in vocoder training,
                    # since we're essentially dealing with a discriminator here.
                    cycle_dist = cycle_dist + torch.nn.functional.l1_loss(out_pred, out_gold.detach())

                train_loss = train_loss + cycle_dist
                cycle_losses_total.append(cycle_dist.item())

        # then we directly update our meta-parameters without
        # the need for any task specific parameters

        train_losses_total.append(train_loss.item())
        l1_losses_total.append(l1_loss.item())
        ssim_losses_total.append(ssim_loss.item())
        mse_losses_total.append(mse_loss.item())
        duration_losses_total.append(duration_loss.item())
        pitch_losses_total.append(pitch_loss.item())
        energy_losses_total.append(energy_loss.item())
        kl_losses_total.append(kl_loss.item())
        if glow_loss is not None:
            glow_losses_total.append(glow_loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        if step % steps_per_checkpoint == 0 and step != 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            net.eval()
            style_embedding_function.eval()
            default_embedding = style_embedding_function(
                batch_of_spectrograms=datasets[0][0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=datasets[0][0][3].unsqueeze(0).to(device)).squeeze()
            print(f"\nTotal Steps: {step}")
            print(f"Total Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            if len(cycle_losses_total) != 0:
                print(f"Cycle Loss: {round(sum(cycle_losses_total) / len(cycle_losses_total), 3)}")
            torch.save({
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scaler"      : grad_scaler.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "step_counter": step,
                "default_emb" : default_embedding,
                },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            path_to_most_recent_plot = plot_progress_spec(net=net,
                                                          device=device,
                                                          lang=lang,
                                                          save_dir=save_directory,
                                                          step=step,
                                                          default_emb=default_embedding)
            if use_wandb:
                wandb.log({
                    "total_loss"   : round(sum(train_losses_total) / len(train_losses_total), 3),
                    "l1_loss"      : round(sum(l1_losses_total) / len(l1_losses_total), 3),
                    "ssim_loss"    : round(sum(ssim_losses_total) / len(ssim_losses_total), 3),
                    "mse_loss"     : round(sum(mse_losses_total) / len(mse_losses_total), 3),
                    "duration_loss": round(sum(duration_losses_total) / len(duration_losses_total), 3),
                    "pitch_loss"   : round(sum(pitch_losses_total) / len(pitch_losses_total), 3),
                    "energy_loss"  : round(sum(energy_losses_total) / len(energy_losses_total), 3),
                    "kl_loss"      : round(sum(kl_losses_total) / len(kl_losses_total), 3),
                    "glow_loss"    : round(sum(glow_losses_total) / len(glow_losses_total), 3),
                    "cycle_loss"   : round(sum(cycle_losses_total) / len(cycle_losses_total), 3) if len(cycle_losses_total) != 0 else 0.0,
                    "Steps"        : step,
                    "progress_plot": wandb.Image(path_to_most_recent_plot)
                    })
            train_losses_total = list()
            cycle_losses_total = list()
            l1_losses_total = list()
            ssim_losses_total = list()
            mse_losses_total = list()
            duration_losses_total = list()
            pitch_losses_total = list()
            energy_losses_total = list()
            kl_losses_total = list()
            glow_losses_total = list()
            net.train()
