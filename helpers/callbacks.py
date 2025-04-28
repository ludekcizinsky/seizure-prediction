from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def get_callbacks(cfg) -> list:
 
    checkpoint_cb = ModelCheckpoint(
        every_n_epochs=cfg.trainer.checkpoint_every_n_epochs,
    ) # Save only the last checkpoint

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_cb, lr_monitor]

    return callbacks