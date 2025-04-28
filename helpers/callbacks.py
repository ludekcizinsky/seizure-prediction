from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

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

def get_callbacks(cfg) -> list:
 
    checkpoint_cb = ModelCheckpoint(
        every_n_epochs=cfg.trainer.checkpoint_every_n_epochs,
    ) # Save only the last checkpoint

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = EpochProgressBar()

    callbacks = [checkpoint_cb, lr_monitor, progress_bar]

    return callbacks