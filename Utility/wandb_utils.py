# Utility function to log the configuration of the training run

import wandb

def record_training_config(batch_size, lang, lr, warmup_steps, fine_tune, phase_1_steps, phase_2_steps):
    wandb.config.batch_size = batch_size
    wandb.config.lang = lang
    wandb.config.lr = lr
    wandb.config.warmup_steps = warmup_steps
    wandb.config.fine_tune = fine_tune
    wandb.config.phase_1_steps = phase_1_steps
    wandb.config.phase_2_steps = phase_2_steps