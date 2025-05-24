import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import f1_score, binary_accuracy

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

    def on_train_epoch_start(self):
        # reset buffers
        self.train_preds = []
        self.train_targets = []

    def training_step(self, batch, batch_idx):

        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float()  # [batch_size, ]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        loss = nn.BCEWithLogitsLoss()(logits, y_batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        preds = (torch.sigmoid(logits) > 0.5).int()
        # store on CPU to avoid OOM
        self.train_preds.append(preds.cpu())
        self.train_targets.append(y_batch.cpu().int())

        return loss

    def on_train_epoch_end(self):
        # concatenate all batches
        preds   = torch.cat(self.train_preds)
        targets = torch.cat(self.train_targets)

        # Compute metrics
        acc = binary_accuracy(preds, targets)
        f1_per_class = f1_score(preds, targets, average=None, num_classes=2, task="multiclass")
        f1_macro = torch.mean(f1_per_class)

        # Log metrics
        self.log("train/acc", acc)
        self.log("train/f1_class_0", f1_per_class[0].item())
        self.log("train/f1_class_1", f1_per_class[1].item())
        self.log("train/f1_macro", f1_macro.item())

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float()  # [batch_size, ]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        loss = nn.BCEWithLogitsLoss()(logits, y_batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        preds = (torch.sigmoid(logits) > 0.5).int()
        # store on CPU to avoid OOM
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y_batch.cpu().int())

    def on_validation_epoch_end(self):
        # concatenate all batches
        preds   = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        # Compute metrics
        acc = binary_accuracy(preds, targets)
        f1_per_class = f1_score(preds, targets, average=None, num_classes=2, task="multiclass")
        f1_macro = torch.mean(f1_per_class)

        # Log metrics
        self.log("val/acc", acc)
        self.log("val/f1_class_0", f1_per_class[0].item())
        self.log("val/f1_class_1", f1_per_class[1].item())
        self.log("val/f1_macro", f1_macro.item())

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