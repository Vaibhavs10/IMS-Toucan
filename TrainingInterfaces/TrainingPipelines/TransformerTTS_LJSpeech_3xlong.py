import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.TransformerTTS.TransformerTTS import Transformer
from TrainingInterfaces.Text_to_Spectrogram.TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TrainingInterfaces.Text_to_Spectrogram.TransformerTTS.transformer_tts_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_3xljspeech as build_path_to_transcript_dict


def run(gpu_id, resume_checkpoint, finetune, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "LJSpeech_3xlong")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "R_v_bt_64_lr_0001_b_46_bce_10_rf_2_3xlong")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len_in_seconds=1,
                                      max_len_in_seconds=35,
                                      rebuild_cache=False)

    model = Transformer(idim=166, odim=80, spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=300000,
               batch_size=14,
               epochs_per_save=10,
               use_speaker_embedding=False,
               lang="en",
               lr=0.0001,
               warmup_steps=8000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune)
