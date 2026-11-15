# MLOps Practice: Paraphrase Detection on GLUE MRPC

An end-to-end MLOps practice project. It fine-tunes a small BERT to decide whether two
sentences are paraphrases (the [GLUE MRPC](https://huggingface.co/datasets/nyu-mll/glue)
task), tracks experiments, exports the model to ONNX, and serves it through a Streamlit
dashboard in a slim Docker image.

```
configs (Hydra) ──▶ train (Lightning + W&B) ──▶ checkpoint ──▶ export (ONNX) ──▶ serve (Streamlit, Docker)
```

| Stage | Tooling |
| --- | --- |
| Data | Hugging Face `datasets`, `transformers` tokenizers |
| Training | PyTorch Lightning, torchmetrics, early stopping and checkpointing |
| Config | Hydra config groups for data, model, trainer, logger and paths |
| Experiment tracking | Weights & Biases (optional) |
| Packaging | ONNX export, `onnxruntime` inference with no torch at serve time |
| Serving | Streamlit dashboard comparing two models side by side, with latency |
| Deployment | Docker image on `python:3.11-slim` with inference dependencies only |

## Project layout

```
├── app/streamlit_app.py     # ONNX inference dashboard
├── configs/                 # Hydra: config.yaml + data/ model/ trainer/ logger/ paths/
├── models/mrpc_model.onnx   # exported model (+ .onnx.data weights), tracked in git
├── src/mrpc/
│   ├── data.py              # MRPCDataModule (LightningDataModule)
│   ├── model.py             # MRPCModel (LightningModule)
│   ├── train.py             # python -m mrpc.train
│   ├── export_onnx.py       # python -m mrpc.export_onnx
│   ├── inference.py         # OnnxPredictor / TorchPredictor + CLI
│   └── evaluate.py          # behavioural test suite (negation, numbers, role reversal, ...)
├── tests/                   # pytest
├── Dockerfile
└── pyproject.toml
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[train,app,dev]"
cp .env.example .env                                   # add your WANDB_API_KEY
```

The committed ONNX model lets you try inference right away:

```bash
python -m mrpc.inference "The firm announced increased earnings." "The company reported higher profits."
streamlit run app/streamlit_app.py
```

## Train

```bash
python -m mrpc.train                         # defaults: model=bert trainer=gpu logger=wandb
python -m mrpc.train trainer=cpu             # CPU
python -m mrpc.train logger.enabled=false    # no W&B account needed
python -m mrpc.train model=roberta data.batch_size=16 trainer.max_epochs=3
```

The best checkpoint is saved to `models/best_checkpoint.ckpt`. Checkpoints, logs and W&B
runs are git-ignored.

## Export and evaluate

```bash
python -m mrpc.export_onnx     # models/best_checkpoint.ckpt -> models/mrpc_model.onnx
python -m mrpc.evaluate        # behavioural suite: per-category pass rates
pytest                         # unit, config and ONNX smoke tests
```

## Docker

```bash
docker build -t mrpc-app .
docker run --rm -p 8501:8501 mrpc-app      # http://localhost:8501
```

The image installs only the inference dependencies and bakes in the tokenizer, so the
container starts without network access. `.env` is excluded through `.dockerignore`.

## Notes

- The default model is [`google/bert_uncased_L-2_H-128_A-2`](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2),
  a 4.4M-parameter BERT. It's fast to iterate on, but don't expect state-of-the-art accuracy.
- `make install | train | export | evaluate | app | test | docker` wraps the commands above.
