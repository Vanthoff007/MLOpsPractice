import torch
import hydra
import logging
from model import mrpcModel
from data import mrpcData

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="configs", config_name="config")
def convert_to_onnx(cfg):
    root_dir = hydra.utils.get_original_cwd()
    logger.info(f"Root directory: {root_dir}")
    model_path = f"{root_dir}/models/best_checkpoint.ckpt"
    logger.info(f"Loading model from: {model_path}")
    mrpc_model = mrpcModel.load_from_checkpoint(model_path).to(device)

    mrpc_data = mrpcData(
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
    )

    mrpc_data.prepare_data()
    mrpc_data.setup()
    input_batch = next(iter(mrpc_data.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0).to(device),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0).to(device),
    }

    logger.info("Converting model to ONNX format...")
    torch.onnx.export(
        mrpc_model,
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),
        f"{root_dir}/models/mrpc_model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
        opset_version=10,
    )

    logger.info("Model successfully converted to ONNX format and saved.")


if __name__ == "__main__":
    convert_to_onnx()
