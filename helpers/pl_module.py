import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm
from torchmetrics.functional.classification import binary_f1_score


class SeizurePredictor(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = model
        self.lr = cfg.optim.lr

    def forward(self, x):
        dtype = next(self.model.parameters()).dtype
        x = x.to(dtype)
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x_batch, y_batch = batch
        x_batch = x_batch  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float()  # [batch_size, ]

        logits = self(x_batch).squeeze(1)  # [batch_size, ]
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

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
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

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

        # Plateau scheduler (will be activated after warmup)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.optim.plateau_factor,
            patience=self.hparams.optim.plateau_patience,
            min_lr=self.hparams.optim.min_lr,
        )

        # Return both schedulers, but only use ReduceLROnPlateau after warmup

        return [optimizer], [
            {
                "scheduler": plateau_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/f1",
                "strict": False,
                "name": "plateau",
            },
        ]
    
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        optimizer.step(closure=optimizer_closure)

        # Warmup for the first N epochs
        warmup_epochs = self.hparams.optim.warmup_epochs
        if self.current_epoch < warmup_epochs:
            lr_scale = float(self.current_epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr  # or self.learning_rate 

    def on_before_optimizer_step(self, optimizer):
        norm_order = 2.0
        norms = grad_norm(self, norm_type=norm_order)
        self.log(
            "Total gradient (norm)",
            norms[f"grad_{norm_order}_norm_total"],
            on_step=False,
            on_epoch=True,
        )
