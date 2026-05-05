import torch
from loguru import logger

_MODEL_NAME = "ProsusAI/finbert"


class FinBERTEmbedder:
    """Local FinBERT inference — no API call, fully offline."""

    def __init__(self, device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        # FinBERT label order: positive=0, negative=1, neutral=2
        self._labels = ["positive", "negative", "neutral"]
        logger.info("FinBERT loaded")

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                results.append({
                    "positive": float(row[0]),
                    "negative": float(row[1]),
                    "neutral": float(row[2]),
                })
        return results

    def score(self, text: str) -> dict:
        return self.score_batch([text])[0]
