from pathlib import Path

import numpy as np
import pytest

from mrpc import LABELS
from mrpc.inference import best, softmax, to_prediction

ONNX_MODEL = Path(__file__).resolve().parents[1] / "models" / "mrpc_model.onnx"


def test_softmax_sums_to_one_and_is_stable():
    probs = softmax(np.array([1000.0, 1001.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert probs[1] > probs[0]


def test_prediction_format():
    pred = to_prediction([0.2, 0.8])
    assert [r["label"] for r in pred] == list(LABELS)
    assert best(pred)["label"] == "paraphrase"


@pytest.mark.network
@pytest.mark.skipif(not ONNX_MODEL.exists(), reason="ONNX model not exported")
def test_onnx_predictor_end_to_end():
    from mrpc.inference import OnnxPredictor

    predictor = OnnxPredictor(str(ONNX_MODEL))
    pred = predictor.predict("Hello world!", "Hello world!")
    assert len(pred) == 2
    assert np.isclose(sum(r["score"] for r in pred), 1.0, atol=1e-5)
