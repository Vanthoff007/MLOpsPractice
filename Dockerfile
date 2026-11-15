# Inference image: Streamlit + onnxruntime only (no torch, no training deps).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

# Bake the tokenizer into the image so the container starts offline.
RUN python -c "from transformers import AutoTokenizer; from mrpc import DEFAULT_MODEL_NAME; AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)"

COPY app ./app
COPY models/mrpc_model.onnx models/mrpc_model.onnx.data ./models/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
