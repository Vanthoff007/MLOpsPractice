import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import mrpcData
from model import mrpcModel

import hydra
from omegaconf.omegaconf import OmegaConf

import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.model_name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer_name}")

    pl.seed_everything(cfg.seed)

    os.makedirs("models", exist_ok=True)

    mrpc_data = mrpcData(
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
    )
    mrpc_model = mrpcModel(model_name=cfg.model.model_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="valid/loss",
        mode="min",
        save_top_k=1,
        filename="best_checkpoint.ckpt",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, mode="min"
    )
    wandb_logger = WandbLogger(
        project=cfg.trainer.project,
        entity=cfg.trainer.entity,
        name=cfg.trainer.name,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(mrpc_model, mrpc_data)


if __name__ == "__main__":
    main()
