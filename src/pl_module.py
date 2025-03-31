import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SeizurePredictor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.loss_fn = ...

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # TODO: adjust this to our dataset
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: adjust the metric computation - we can use torch metrics
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
