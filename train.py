from datetime import datetime
import os

import hydra
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from helpers.callbacks import get_callbacks
from helpers.dataset import get_dataloaders
from helpers.pl_module import SeizurePredictor
from helpers.models.constructor import ModulardModel


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    print("-" * 50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-" * 50)

    L.seed_everything(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    if not cfg.debug:
        logger = WandbLogger(
            project=cfg.logger.project,
            save_dir=cfg.output_dir,
            log_model="all",
            tags=cfg.logger.tags,
        )
    else:
        run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="debug",
            version=run_version,
        )

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    model = ModulardModel(cfg)
    pl_module = SeizurePredictor(cfg, model)
    callbacks = get_callbacks(cfg)

    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        precision=cfg.trainer.precision,
        enable_progress_bar=False,
        log_every_n_steps=len(trn_dataloader),
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
