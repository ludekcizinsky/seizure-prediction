import sys
from datetime import datetime
import os
import traceback


import hydra
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from helpers.callbacks import get_callbacks
from helpers.dataset import get_dataloaders
from helpers.pl_module import SeizurePredictor
from helpers.utils import run_eval_and_save_submission

import wandb

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    L.seed_everything(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    if not cfg.debug:
        cfg.launch_cmd = " ".join(sys.argv)
        logger = WandbLogger(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            save_dir=cfg.output_dir,
            log_model="all",
            tags=cfg.logger.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="debug",
            version=run_version,
        )

    print("-" * 50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-" * 50)

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    pl_module = SeizurePredictor(cfg)
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

    try:
        trainer.fit(pl_module, trn_dataloader, val_dataloader)
        run_eval_and_save_submission(trainer, cfg)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("-" * 50)
    finally:
        wandb.finish()

if __name__ == "__main__":
    train()
