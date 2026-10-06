"""Single source of truth for every tunable in the project."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "reviews.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

LABELS = {0: "Negative", 1: "Positive"}


@dataclass(frozen=True)
class Config:
    # Vectorization (blueprint step 3)
    ngram_range: tuple = (1, 2)       # unigrams + bigrams -> captures "not good"
    max_features: int = 10_000        # cap dimensionality
    min_df: int = 2                   # drop rare typos
    sublinear_tf: bool = True         # log-scale TF, dampens spammy repetition

    # Naive Bayes (blueprint step 5)
    alpha: float = 1.0                # Laplace smoothing
    imbalance_threshold: float = 0.30  # minority share below this -> ComplementNB

    # Training
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
