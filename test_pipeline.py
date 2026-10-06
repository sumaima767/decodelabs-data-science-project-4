import pytest
from scipy import sparse

from sentiment_nlp.model import build_pipeline, choose_classifier
from sentiment_nlp.preprocessing import (NEGATIONS, build_stopwords, preprocess_text,
                                         treebank_to_wordnet)


def test_negations_survive_stopword_filter():
    assert not (NEGATIONS & build_stopwords())
    assert "not" in preprocess_text("I am not happy.").split()


def test_contractions_become_negations():
    assert preprocess_text("It didn't work") == "not work"


def test_pos_guided_lemmatization():
    assert preprocess_text("He went home").split()[0] == "go"
    assert treebank_to_wordnet("VBD") == "v"
    assert treebank_to_wordnet("XYZ") == "n"  # safe default


def test_html_urls_punctuation_stripped():
    out = preprocess_text("TERRIBLE!!! <br> see http://x.com")
    assert out == "terrible see" or out == "terrible"
    assert "<" not in out and "!" not in out


def test_classifier_selection():
    assert choose_classifier([0] * 50 + [1] * 50) == "multinomial"
    assert choose_classifier([0] * 5 + [1] * 95) == "complement"


@pytest.mark.parametrize("clf", ["multinomial", "complement"])
def test_pipeline_outputs_sparse_csr(clf):
    X = ["great product", "terrible product", "great value", "terrible quality"] * 3
    y = [1, 0, 1, 0] * 3
    pipe = build_pipeline(clf).fit(X, y)
    mat = pipe[:-1].transform(X)
    assert sparse.isspmatrix_csr(mat)
    assert pipe.predict(["great"])[0] == 1
