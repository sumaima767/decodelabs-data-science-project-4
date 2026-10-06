"""Inference.

    python -m sentiment_nlp.predict "I am not happy with this phone"
"""
import argparse
from typing import Dict, List, Sequence

import joblib

from . import config


class SentimentModel:
    def __init__(self, path=config.MODEL_PATH):
        try:
            self.pipe = joblib.load(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No trained model at {path}. Run `python -m sentiment_nlp.train` first.") from e

    def predict(self, texts: Sequence[str]) -> List[Dict]:
        proba = self.pipe.predict_proba(list(texts))
        return [
            {"text": t, "label": config.LABELS[int(p.argmax())], "confidence": round(float(p.max()), 3)}
            for t, p in zip(texts, proba)
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    for r in SentimentModel().predict(ap.parse_args().texts):
        print(f"{r['label']:<9} ({r['confidence']:.0%})  {r['text']}")
