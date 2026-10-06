"""Ingest + validate the review dataset (CSV with `text` and `label` columns)."""
from pathlib import Path

import pandas as pd

_LABEL_MAP = {"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0}


def load_reviews(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/make_sample_data.py` or supply your own CSV."
        )
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(_LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Labels must be positive/negative (or 1/0).")
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("Need both Positive and Negative examples to train.")
    return df
