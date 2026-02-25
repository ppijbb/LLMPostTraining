from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class RewardModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Reward models are typically sequence classification models for scoring
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score(self, texts: list[str]):
        # The reward model usually takes a pair of texts (e.g., prompt and response)
        # or a single text (response) to score its quality.
        # For simplicity, we assume it scores a single generated text.
        # In a real scenario, you might need to concatenate prompt and response.
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Assuming a single logit output for score
        scores = outputs.logits.squeeze(-1).tolist()
        return scores 