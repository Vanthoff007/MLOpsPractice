"""Train with Hydra config: `python -m mrpc.train trainer=cpu model=bert`."""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mrpc.data import MRPCDataModule
from mrpc.model import MRPCModel

log = logging.getLogger(__name__)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def build_logger(cfg: DictConfig):
    if not cfg.logger.enabled:
        return False
    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.logger.run_name,
        save_dir=cfg.paths.log_dir,
    )


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = MRPCDataModule(
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    model = MRPCModel(model_name=cfg.model.model_name, lr=cfg.model.lr)

    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            filename="best_checkpoint",
            monitor="valid/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="valid/loss", patience=cfg.early_stopping_patience, mode="min"),
    ]

    trainer = pl.Trainer(
        logger=build_logger(cfg),
        callbacks=callbacks,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )
    trainer.fit(model, data)
    log.info("Best checkpoint: %s", callbacks[0].best_model_path)


if __name__ == "__main__":
    main()
