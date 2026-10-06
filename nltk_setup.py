"""Idempotent NLTK resource bootstrap, so a fresh machine just works."""
import logging

import nltk

log = logging.getLogger(__name__)

# package name -> candidate lookup paths (corpora may be stored zipped)
_RESOURCES = {
    "stopwords": ["corpora/stopwords"],
    "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
    "averaged_perceptron_tagger_eng": ["taggers/averaged_perceptron_tagger_eng"],
}


def _present(paths) -> bool:
    for p in paths:
        try:
            nltk.data.find(p)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources() -> None:
    for package, paths in _RESOURCES.items():
        if not _present(paths):
            log.info("Downloading NLTK resource: %s", package)
            if not nltk.download(package, quiet=True) or not _present(paths):
                raise RuntimeError(
                    f"Could not obtain NLTK resource '{package}'. "
                    "Check your network or run: python -m nltk.downloader " + package
                )
