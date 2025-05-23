import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_f1_score

from helpers.models.constructor import ModulardModel

class SeizurePredictor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ModulardModel(cfg)
        self.lr = cfg.optim.lr

    def forward(self, x):
        x = x.to(next(self.model.parameters()).dtype)
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float()  # [batch_size, ]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        loss = nn.BCEWithLogitsLoss()(logits, y_batch)

        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = (preds == y_batch.int()).float().mean()
        f1 = binary_f1_score(preds, y_batch.int())
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float()  # [batch_size, ]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        loss = nn.BCEWithLogitsLoss()(logits, y_batch)

        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = (preds == y_batch.int()).float().mean()
        f1 = binary_f1_score(preds, y_batch.int())
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        preds = (torch.sigmoid(logits) > 0.5).int()

        return {
            "y_batch": y_batch.int() if type(y_batch) == torch.Tensor else y_batch,
            "preds_batch": preds,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.optim.weight_decay,
        )

        return optimizer