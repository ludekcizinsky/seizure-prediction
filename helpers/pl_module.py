import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.utilities import grad_norm
from torchmetrics.functional.classification import binary_f1_score


class SeizurePredictor(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = model
        self.lr = cfg.optim.lr
        distance_file = cfg.model.get("distance_file",None)
        if distance_file is not None:
            distance_matrix = torch.tensor(pd.read_csv(distance_file).pivot(index='from', columns='to', values='distance').to_numpy(),device='cuda:0',dtype=torch.float32)
            adjacency = (distance_matrix <= cfg.model.get("distance_thresh",None)).int()
            self.edge_index = torch.argwhere(adjacency==1).transpose(-1,-2).to(torch.long)
        else:
            self.edge_index = []

    def forward(self, x):
        dtype = next(self.model.parameters()).dtype
        x = x.to(dtype)
        if self.edge_index != []:
            return self.model(x, self.edge_index)
        else:
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.optim.weight_decay,
        )

        return optimizer