import pytorch_lightning as pl

from src.helpers import prepare_wandb_logger
from src.helpers import prepare_callbacks
from src.helpers import prepare_data_loaders
from src.helpers import prepare_pl_module

def main():

    cfg = ... # todo: here we ideally read the config file such that we can then do cfg.model_name etc.

    wandb_logger = prepare_wandb_logger(cfg)
    callbacks = prepare_callbacks(cfg)
    trn_loader, val_loader = prepare_data_loaders(cfg)
    pl_module = prepare_pl_module(cfg)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(pl_module, trn_loader, val_loader)

if __name__ == "__main__":
    main()
