import hydra
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from helpers.callbacks import get_callbacks
from helpers.dataset import get_dataloaders
from helpers.pl_module import SeizurePredictor


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    print("-" * 50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-" * 50)

    L.seed_everything(cfg.seed)

    if not cfg.debug:
        logger = WandbLogger(
            project=cfg.logger.project,
            save_dir=cfg.output_dir,
            log_model="all",
            tags=cfg.logger.tags,
        )
    else:
        logger = None

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    model = hydra.utils.instantiate(cfg.model)
    pl_module = SeizurePredictor(cfg, model)
    callbacks = get_callbacks(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.trainer.grad_clip,
        deterministic=True,
        precision=cfg.trainer.precision,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)


if __name__ == "__main__":
    train()
