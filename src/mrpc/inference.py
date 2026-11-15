"""Predictors for paraphrase detection.

`OnnxPredictor` needs only onnxruntime + a tokenizer (no torch), so it is what
the Streamlit app and Docker image use. `TorchPredictor` runs a Lightning
checkpoint directly.
"""

from typing import Protocol

import numpy as np
from transformers import AutoTokenizer

from mrpc import DEFAULT_MODEL_NAME, LABELS

Prediction = list[dict]


class Predictor(Protocol):
    def predict(self, sentence1: str, sentence2: str) -> Prediction: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def to_prediction(probs) -> Prediction:
    return [{"label": label, "score": float(p)} for label, p in zip(LABELS, probs)]


def best(prediction: Prediction) -> dict:
    return max(prediction, key=lambda r: r["score"])


class OnnxPredictor:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 512,
    ):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def predict(self, sentence1: str, sentence2: str) -> Prediction:
        enc = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(None, inputs)[0][0]
        return to_prediction(softmax(logits))


class TorchPredictor:
    def __init__(self, checkpoint_path: str, max_length: int = 512):
        import torch

        from mrpc.model import MRPCModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MRPCModel.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.hparams.model_name)
        self.max_length = max_length

    def predict(self, sentence1: str, sentence2: str) -> Prediction:
        enc = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = self.torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return to_prediction(probs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score one sentence pair.")
    parser.add_argument("sentence1")
    parser.add_argument("sentence2")
    parser.add_argument("--onnx", default="models/mrpc_model.onnx")
    parser.add_argument("--checkpoint", help="Use a Lightning checkpoint instead of ONNX")
    args = parser.parse_args()

    predictor = (
        TorchPredictor(args.checkpoint) if args.checkpoint else OnnxPredictor(args.onnx)
    )
    print(predictor.predict(args.sentence1, args.sentence2))
