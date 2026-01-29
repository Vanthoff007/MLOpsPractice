import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from data import mrpcData


class OnnxInference:
    def __init__(self, model_path):
        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.processor = mrpcData()
        self.labels = ["not_paraphrase", "paraphrase"]

    def predict(self, sentence1, sentence2):
        inference_data = {"sentence1": sentence1, "sentence2": sentence2}
        tokenizd_data = self.processor.tokenize_data(inference_data)

        inputs = {
            "input_ids": np.expand_dims(tokenizd_data["input_ids"], axis=0).astype(
                np.int64
            ),
            "attention_mask": np.expand_dims(
                tokenizd_data["attention_mask"], axis=0
            ).astype(np.int64),
        }

        outputs = self.ort_session.run(None, inputs)
        scores = softmax(outputs[0][0])
        preds = []
        for score, label in zip(scores, self.labels):
            preds.append({"label": label, "score": float(score)})
        return preds


if __name__ == "__main__":
    sentence1 = "The firm announced increased earnings."
    sentence2 = "The company reported higher profits."
    predictor = OnnxInference("./models/mrpc_model.onnx")
    print(predictor.predict(sentence1=sentence1, sentence2=sentence2))
