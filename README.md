# NLP & Sentiment Analysis Engine (DecodeLabs Project 4)

Raw review text -> clean tokens -> sparse TF-IDF matrix -> Naive Bayes -> Positive / Negative.

## Architecture

```
sentiment-nlp/
├── sentiment_nlp/
│   ├── config.py         # all hyper-parameters + paths (one place to tune)
│   ├── nltk_setup.py     # auto-downloads stopwords / wordnet / POS tagger
│   ├── preprocessing.py  # normalize -> tokenize -> POS-tag -> stop-words -> lemmatize
│   ├── data.py           # CSV ingest + validation
│   ├── model.py          # Pipeline factory + MultinomialNB/ComplementNB selection
│   ├── train.py          # split, cross-validate, evaluate, persist
│   └── predict.py        # SentimentModel class + CLI
├── scripts/make_sample_data.py   # synthetic demo data
├── tests/test_pipeline.py
├── data/reviews.csv
└── models/               # sentiment_model.joblib, metrics.json
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py          # or drop in your own data/reviews.csv (text,label)
python -m sentiment_nlp.train
python -m sentiment_nlp.predict "I am not happy with this phone"
pytest
```

## Key decisions (mapped to the blueprint)

| Blueprint step | Implementation |
|---|---|
| 1. Stop-words | NLTK list **minus** `NEGATIONS` (set difference). Contractions like `didn't` are expanded to `did not` first so the negation survives tokenization. |
| 2. Lemmatization | `WordNetLemmatizer` with Treebank -> WordNet POS mapping. POS tagging runs *before* stop-word removal so the tagger sees the full sentence. |
| 3. Vectorization | `TfidfVectorizer(ngram_range=(1,2), max_features=10_000, min_df=2, sublinear_tf=True)` |
| 4. Memory | TF-IDF output is a SciPy **CSR** matrix; nothing ever calls `.toarray()`. A test asserts this. |
| 5. Inference | `MultinomialNB(alpha=1.0)` (Laplace); auto-switches to `ComplementNB` if the minority class is < 30%. |

Preprocessing is a scikit-learn transformer *inside* the Pipeline, so cross-validation has no leakage
and the saved model applies identical cleaning at inference time.

## Caveat

`data/reviews.csv` is **synthetic** (template-generated) purely so the project runs offline.
Its ~99% score is not meaningful. Train on a real dataset (IMDB, Amazon, Yelp) for real numbers.
