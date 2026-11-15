from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.mark.parametrize("model", ["bert", "roberta"])
@pytest.mark.parametrize("trainer", ["cpu", "gpu"])
def test_every_config_combination_composes(model, trainer):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose("config", overrides=[f"model={model}", f"trainer={trainer}"])
    assert cfg.model.model_name
    assert cfg.paths.checkpoint.endswith(".ckpt")
    # Trainer group must contain only pytorch_lightning.Trainer kwargs.
    assert "project" not in cfg.trainer
