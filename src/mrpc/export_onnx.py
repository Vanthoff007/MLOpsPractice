"""Export a trained checkpoint to ONNX: `python -m mrpc.export_onnx`."""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from mrpc.model import MRPCModel
from mrpc.train import CONFIG_DIR

log = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt = Path(cfg.paths.checkpoint)
    out = Path(cfg.paths.onnx_model)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading checkpoint %s", ckpt)
    model = MRPCModel.load_from_checkpoint(ckpt, map_location="cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
    sample = tokenizer(
        "The firm announced increased earnings.",
        "The company reported higher profits.",
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    log.info("Saved ONNX model to %s", out)


if __name__ == "__main__":
    main()
