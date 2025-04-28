from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import torch

def get_callbacks(cfg):
 
    checkpoint_cb = ModelCheckpoint(
        every_n_epochs=cfg.trainer.checkpoint_every_n_epochs,
    ) # Save only the last checkpoint

    progress_bar = EpochProgressBar()
    lr_scheduler = WarmupPlateauScheduler(cfg)

    callbacks = [checkpoint_cb, progress_bar, lr_scheduler]

    return callbacks


class EpochProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_train_start(self, trainer, pl_module):
        # Create a standard tqdm bar with total = num_epochs
        self.pbar = tqdm(
            total=trainer.max_epochs,
            desc=f"Epoch 0/{trainer.max_epochs}",
            leave=True,
            dynamic_ncols=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        # Increment the bar by 1
        self.pbar.update(1)
        # Update the description in-place
        self.pbar.set_description(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        # (no new lines, just redraw this line)
    
    def on_train_end(self, trainer, pl_module):
        # Close it cleanly
        self.pbar.close()


class WarmupPlateauScheduler(Callback):
    def __init__(self, cfg):
        """
        Args:
            warmup_epochs: number of epochs to linearly increase LR from 0 → base_lr
            plateau_kwargs: kwargs for torch.optim.lr_scheduler.ReduceLROnPlateau
                            (e.g. {'mode':'min','patience':2,'factor':0.1,'min_lr':1e-6})
        """
        super().__init__()
        self.cfg = cfg

    def on_train_start(self, trainer, pl_module):
        # Grab the (single) optimizer
        opt = trainer.optimizers[0]

        # 1) Warmup: LR = base_lr * ((epoch+1)/warmup_epochs), capped at 1.0
        warmup_epochs = self.cfg.optim.warmup_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1.0)
        )
        # 2) Plateau: starts having effect only after warmup
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.cfg.optim.plateau_patience,
            factor=self.cfg.optim.plateau_factor,
            min_lr=self.cfg.optim.min_lr,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # after warmup_epochs, switch to plateau; before that, do warmup.step()
        warmup_epochs = self.cfg.optim.warmup_epochs
        if epoch < warmup_epochs:
            self.warmup_scheduler.step()

        opt = trainer.optimizers[0]
        lr = opt.param_groups[0]["lr"]
        pl_module.log("optim/lr", lr, on_epoch=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        warmup_epochs = self.cfg.optim.warmup_epochs
        if epoch >= warmup_epochs:
            self.plateau_scheduler.step(metrics["val/f1"])
