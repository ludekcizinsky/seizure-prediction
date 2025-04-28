import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeizurePredictor(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.unsqueeze(1)  # [batch_size, 1]

        logits = self(x_batch)  # [batch_size, 1]
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y_batch).float().mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.unsqueeze(1)  # [batch_size, 1]

        logits = self(x_batch)  # [batch_size, 1]
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y_batch).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.trainer.max_epochs
        )
        return [optimizer], [scheduler]
