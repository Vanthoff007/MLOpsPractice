"""Streamlit dashboard: compare paraphrase predictions from two ONNX models.

Run: `streamlit run app/streamlit_app.py`
"""

import os
import time

import streamlit as st

from mrpc import DEFAULT_MODEL_NAME
from mrpc.inference import OnnxPredictor, best

# Add more models here: display name -> (ONNX path, tokenizer name)
MODEL_REGISTRY = {
    "MRPC BERT (tiny)": (
        os.getenv("MRPC_ONNX_MODEL", "models/mrpc_model.onnx"),
        DEFAULT_MODEL_NAME,
    ),
}

st.set_page_config(page_title="Paraphrase Detection", layout="centered")
st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 2rem; }
    .model-title { font-size: 20px; font-weight: 600; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(name: str) -> OnnxPredictor:
    path, tokenizer = MODEL_REGISTRY[name]
    return OnnxPredictor(path, tokenizer_name=tokenizer)


def predict(name: str, s1: str, s2: str):
    predictor = load_model(name)
    start = time.perf_counter()
    results = predictor.predict(s1, s2)
    latency_ms = (time.perf_counter() - start) * 1000
    return results, best(results), latency_ms


def render(column, name: str, s1: str, s2: str) -> None:
    results, top, latency_ms = predict(name, s1, s2)
    with column:
        st.markdown(f'<div class="model-title">{name}</div>', unsafe_allow_html=True)
        message = f"{top['label']} ({top['score'] * 100:.2f}%)"
        (st.success if top["label"] == "paraphrase" else st.info)(message)
        st.caption(f"Inference latency: {latency_ms:.2f} ms")
        st.write("Confidence")
        for r in results:
            st.write(r["label"])
            st.progress(int(r["score"] * 100))


st.title("Paraphrase Detection")
st.caption("ONNX inference dashboard · GLUE MRPC")

with st.form("inference_form"):
    sentence1 = st.text_input("Sentence 1", "The firm announced increased earnings.")
    sentence2 = st.text_input("Sentence 2", "The company reported higher profits.")
    col1, col2 = st.columns(2)
    model_a = col1.selectbox("Model A", list(MODEL_REGISTRY))
    model_b = col2.selectbox("Model B", list(MODEL_REGISTRY))
    submitted = st.form_submit_button("Run inference")

if submitted:
    if sentence1.strip() and sentence2.strip():
        col_a, col_b = st.columns(2)
        render(col_a, model_a, sentence1, sentence2)
        render(col_b, model_b, sentence1, sentence2)
    else:
        st.warning("Please enter both sentences.")
