"""Paraphrase detection on GLUE MRPC: training, ONNX export and inference."""

DEFAULT_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
LABELS = ("not_paraphrase", "paraphrase")

__all__ = ["DEFAULT_MODEL_NAME", "LABELS"]
