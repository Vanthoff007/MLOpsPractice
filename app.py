import time
import streamlit as st
from inference_onnx import OnnxInference

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Paraphrase Detection Dashboard",
    layout="centered",
)

# --------------------------------------------------
# Styling (clean HF-style dark UI)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 2rem;
    }

    .card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #2a2a2a;
        background-color: #0e1117;
    }

    .model-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .latency {
        font-size: 13px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Model registry
# --------------------------------------------------
MODEL_REGISTRY = {
    "MRPC BERT": "./models/mrpc_model.onnx",
    # Add more models here
    # "RoBERTa": "./models/roberta.onnx",
}


# --------------------------------------------------
# Cached model loader
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return OnnxInference(path)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Paraphrase Detection")
st.caption("ONNX inference dashboard")

# --------------------------------------------------
# Input form
# --------------------------------------------------
with st.form("inference_form"):
    sentence1 = st.text_input("Sentence 1")
    sentence2 = st.text_input("Sentence 2")

    col1, col2 = st.columns(2)
    model_a = col1.selectbox("Model A", list(MODEL_REGISTRY.keys()))
    model_b = col2.selectbox("Model B", list(MODEL_REGISTRY.keys()))

    submitted = st.form_submit_button("Run Inference")


# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
def run_prediction(model_name, s1, s2):
    predictor = load_model(MODEL_REGISTRY[model_name])

    start = time.time()
    results = predictor.predict(s1, s2)
    latency = (time.time() - start) * 1000

    best = max(results, key=lambda x: x["score"])
    return results, best, latency


# --------------------------------------------------
# Results
# --------------------------------------------------
if submitted:
    if sentence1.strip() and sentence2.strip():
        colA, colB = st.columns(2)

        results_a, best_a, latency_a = run_prediction(model_a, sentence1, sentence2)
        results_b, best_b, latency_b = run_prediction(model_b, sentence1, sentence2)

        # ---------------- Model A ----------------
        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="model-title">{model_a}</div>',
                unsafe_allow_html=True,
            )

            if best_a["label"] == "paraphrase":
                st.success(f"{best_a['label']} ({best_a['score'] * 100:.2f}%)")
            else:
                st.info(f"{best_a['label']} ({best_a['score'] * 100:.2f}%)")

            st.caption(f"Inference latency: {latency_a:.2f} ms")

            st.write("Confidence")
            for r in results_a:
                st.write(r["label"])
                st.progress(int(r["score"] * 100))

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Model B ----------------
        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="model-title">{model_b}</div>',
                unsafe_allow_html=True,
            )

            if best_b["label"] == "paraphrase":
                st.success(f"{best_b['label']} ({best_b['score'] * 100:.2f}%)")
            else:
                st.info(f"{best_b['label']} ({best_b['score'] * 100:.2f}%)")

            st.caption(f"Inference latency: {latency_b:.2f} ms")

            st.write("Confidence")
            for r in results_b:
                st.write(r["label"])
                st.progress(int(r["score"] * 100))

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter both sentences.")
