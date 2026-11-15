"""GLUE MRPC data module."""

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from mrpc import DEFAULT_MODEL_NAME

COLUMNS = ["input_ids", "attention_mask", "label"]


class MRPCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        load_dataset("glue", "mrpc")

    def tokenize(self, batch):
        return self.tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        dataset = load_dataset("glue", "mrpc")
        self.train_data = dataset["train"].map(self.tokenize, batched=True)
        self.val_data = dataset["validation"].map(self.tokenize, batched=True)
        self.train_data.set_format(type="torch", columns=COLUMNS)
        self.val_data.set_format(type="torch", columns=COLUMNS)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    dm = MRPCDataModule()
    dm.prepare_data()
    dm.setup()
    print(next(iter(dm.train_dataloader()))["input_ids"].shape)
