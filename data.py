import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class mrpcData(pl.LightningDataModule):
    def __init__(
        self,
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=32,
        max_length=512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        load_dataset("glue", "mrpc")

    def tokenize_data(self, batch):
        return self.tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        dataset = load_dataset("glue", "mrpc")
        self.train_data = dataset["train"].map(self.tokenize_data, batched=True)
        self.val_data = dataset["validation"].map(self.tokenize_data, batched=True)

        self.train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        self.val_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)


if __name__ == "__main__":
    data_model = mrpcData()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
