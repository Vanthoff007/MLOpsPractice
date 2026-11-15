.PHONY: install train export evaluate app test docker

install:
	pip install -e ".[train,app,dev]"

train:
	python -m mrpc.train

export:
	python -m mrpc.export_onnx

evaluate:
	python -m mrpc.evaluate

app:
	streamlit run app/streamlit_app.py

test:
	pytest

docker:
	docker build -t mrpc-app . && docker run --rm -p 8501:8501 mrpc-app
