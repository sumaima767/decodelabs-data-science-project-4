"""Train, evaluate and persist the model.

    python -m sentiment_nlp.train --data data/reviews.csv
"""
import argparse
import json
import logging

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config
from .data import load_reviews
from .model import build_pipeline, choose_classifier

log = logging.getLogger("train")


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=config.DATA_PATH)
    p.add_argument("--classifier", choices=["auto", "multinomial", "complement"], default="auto")
    p.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = config.Config()
    df = load_reviews(args.data)
    log.info("Loaded %d reviews", len(df))

    # Stratified split keeps the class ratio identical in train and test.
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=cfg.test_size,
        stratify=df["label"], random_state=cfg.random_state,
    )

    clf_name = choose_classifier(y_tr, cfg) if args.classifier == "auto" else args.classifier
    pipe = build_pipeline(clf_name, cfg)

    metrics = {"classifier": clf_name, "n_train": len(X_tr), "n_test": len(X_te)}
    if not args.no_cv:
        cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        metrics["cv_f1_macro_mean"], metrics["cv_f1_macro_std"] = float(scores.mean()), float(scores.std())
        log.info("CV macro-F1: %.3f +/- %.3f", scores.mean(), scores.std())

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics.update(
        accuracy=float(accuracy_score(y_te, pred)),
        f1_macro=float(f1_score(y_te, pred, average="macro")),
        confusion_matrix=confusion_matrix(y_te, pred).tolist(),
        vocabulary_size=len(pipe.named_steps["tfidf"].vocabulary_),
    )
    print(classification_report(y_te, pred, target_names=list(config.LABELS.values())))

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Saved model -> %s", config.MODEL_PATH)
    return metrics


if __name__ == "__main__":
    main()
