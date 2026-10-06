"""Strict text pre-processing: clean -> tokenize -> POS-tag -> drop stop-words -> lemmatize.

Exposed as a scikit-learn transformer so it lives *inside* the Pipeline.
That guarantees training and inference use byte-identical cleaning.
"""
import re
from functools import lru_cache
from typing import Iterable, List, Optional

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

from .nltk_setup import ensure_nltk_resources

# Words that flip sentiment. NLTK's default list contains some of these,
# and deleting them turns "I am not happy" into "I am happy".
NEGATIONS = frozenset({"no", "nor", "not", "never", "cannot", "none", "nothing", "neither"})

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTRACTIONS = (
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),   # didn't, isn't, wasn't, couldn't ...
)
_TOKENIZER = RegexpTokenizer(r"[a-z]+")  # letters only: drops digits, "!!!", leftovers


@lru_cache(maxsize=1)
def build_stopwords() -> frozenset:
    """NLTK English stop-words minus negations (set difference)."""
    ensure_nltk_resources()
    return frozenset(stopwords.words("english")) - NEGATIONS


def treebank_to_wordnet(tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet's 4 categories (default: noun)."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(
        tag[:1], wordnet.NOUN
    )


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_resources()
    return WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str, pos: str) -> str:
    return _lemmatizer().lemmatize(token, pos)


def normalize(text: str) -> str:
    """Character normalization: HTML/URLs out, lowercase, contractions expanded."""
    text = _HTML.sub(" ", str(text))
    text = _URL.sub(" ", text)
    text = text.lower().replace("\u2019", "'")
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def preprocess_text(text: str, stop: Optional[frozenset] = None) -> str:
    stop = build_stopwords() if stop is None else stop
    tokens = _TOKENIZER.tokenize(normalize(text))
    # Tag BEFORE removing stop-words: the tagger needs the full sentence for context.
    out: List[str] = []
    for token, tag in pos_tag(tokens):
        if token in stop or len(token) < 2:
            continue
        out.append(_lemma(token, treebank_to_wordnet(tag)))
    return " ".join(out)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless transformer: iterable of raw strings -> list of cleaned strings."""

    def fit(self, X, y=None):
        ensure_nltk_resources()
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        stop = build_stopwords()
        return [preprocess_text(t, stop) for t in X]
