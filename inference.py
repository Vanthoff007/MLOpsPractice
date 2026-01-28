import torch
from model import mrpcModel
from transformers import AutoTokenizer


class Inference:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = model_path
        self.model = mrpcModel.load_from_checkpoint(model_path).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.hparams.model_name)
        self.labels = ["not_paraphrase", "paraphrase"]

    def predict(self, sentence1, sentence2):
        inference_data = {"sentence1": sentence1, "sentence2": sentence2}
        tokenized_data = self.tokenizer(
            inference_data["sentence1"],
            inference_data["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        tokenized_data = {k: v.to(self.device) for k, v in tokenized_data.items()}

        with torch.no_grad():
            logits = self.model(
                tokenized_data["input_ids"],
                tokenized_data["attention_mask"],
            )
            probs = torch.softmax(logits, dim=-1)[0]

        return [
            {"label": label, "score": float(score)}
            for label, score in zip(self.labels, probs)
        ]


if __name__ == "__main__":
    sentence1 = "The firm announced increased earnings."
    sentence2 = "The company reported higher profits."
    predictor = Inference("./models/mrpc-epoch=04-val_loss=0.5806.ckpt")
    print(predictor.predict(sentence1=sentence1, sentence2=sentence2))
