"""Pipeline factory: Preprocess -> TF-IDF (SciPy CSR) -> Naive Bayes."""
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from .config import Config
from .preprocessing import TextPreprocessor

log = logging.getLogger(__name__)


def choose_classifier(y, cfg: Config = Config()) -> str:
    """MultinomialNB for balanced data, ComplementNB when the minority class is rare."""
    minority_share = np.bincount(np.asarray(y)).min() / len(y)
    name = "complement" if minority_share < cfg.imbalance_threshold else "multinomial"
    log.info("Minority class share %.1f%% -> %s NB", minority_share * 100, name)
    return name


def build_pipeline(classifier: str = "multinomial", cfg: Config = Config()) -> Pipeline:
    nb = {"multinomial": MultinomialNB, "complement": ComplementNB}[classifier]
    return Pipeline(
        [
            ("preprocess", TextPreprocessor()),
            (
                "tfidf",
                # Emits a scipy.sparse CSR matrix: zeros are never stored.
                TfidfVectorizer(
                    ngram_range=cfg.ngram_range,
                    max_features=cfg.max_features,
                    min_df=cfg.min_df,
                    sublinear_tf=cfg.sublinear_tf,
                ),
            ),
            ("clf", nb(alpha=cfg.alpha)),  # alpha=1.0 -> Laplace smoothing
        ]
    )
