"""Lightning module wrapping a Hugging Face sequence classifier."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import AutoModelForSequenceClassification

from mrpc import DEFAULT_MODEL_NAME


class MRPCModel(pl.LightningModule):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, lr: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        # Attribute name kept as `bert` so older checkpoints still load.
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, input_ids, attention_mask):
        """Return raw logits, shape (batch, 2). Kept tensor-only for ONNX export."""
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    def _step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        return loss, logits.argmax(dim=1)

    def training_step(self, batch, batch_idx):
        loss, preds = self._step(batch)
        self.train_acc(preds, batch["label"])
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self._step(batch)
        labels = batch["label"]
        for metric in (self.val_acc, self.val_precision, self.val_recall, self.val_f1):
            metric(preds, labels)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        self.log("valid/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("valid/precision", self.val_precision, on_epoch=True)
        self.log("valid/recall", self.val_recall, on_epoch=True)
        self.log("valid/f1", self.val_f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
