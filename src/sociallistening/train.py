"""Train and evaluate tweet sentiment classifiers.

Run:  python -m sociallistening.train

What it does
1. Cleans the 10,000 labelled tweets and makes a stratified 80/20 train/test split.
2. Compares two TF-IDF models, Logistic Regression and Multinomial Naive Bayes,
   with 5-fold cross-validation on the training set.
3. Evaluates both on the held-out test set (precision, recall, F1, confusion matrix).
4. Runs McNemar's test to check whether the difference between the two models on
   the same test tweets is statistically significant.
5. Measures label leakage by retraining Logistic Regression with emoticons kept.
6. Saves the best model to models/ and the metrics and charts to reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import load_labelled
from .preprocess import clean_text

SEED = 42
LABELS = ["negative", "positive"]


def make_vectorizer() -> TfidfVectorizer:
    # Unigrams + bigrams so phrases like "not good" are captured.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("tfidf", make_vectorizer()), ("clf", LogisticRegression(C=2.0, max_iter=2000))]
        ),
        "naive_bayes": Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB(alpha=0.5))]),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions.

    H0: both models have the same error rate on this test set. Only tweets where the
    models disagree carry information: b = A right / B wrong, c = A wrong / B right.
    Under H0, b ~ Binomial(b + c, 0.5).
    """
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    p_value = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p_value)}


def top_features(model: Pipeline, n: int = 15) -> dict[str, list[str]]:
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    return {"negative": vocab[order[:n]].tolist(), "positive": vocab[order[-n:][::-1]].tolist()}


def save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(models_dir: Path, reports_dir: Path) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_labelled()
    df["clean"] = df["text"].map(clean_text)
    df["clean_with_emoticons"] = df["text"].map(lambda t: clean_text(t, keep_emoticons=True))
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    train, test = df.loc[train_idx], df.loc[test_idx]
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results: dict = {
        "dataset": {
            "tweets_used": int(len(df)),
            "train": int(len(train)),
            "test": int(len(test)),
            "positive_share": round(float(df["label"].mean()), 3),
        },
        "models": {},
    }
    predictions = {}
    fitted = {}

    for name, model in build_models().items():
        cv_f1 = cross_val_score(model, train["clean"], y_train, cv=cv, scoring="f1_macro")
        model.fit(train["clean"], y_train)
        pred = model.predict(test["clean"])
        predictions[name], fitted[name] = pred, model
        report = classification_report(y_test, pred, target_names=LABELS, output_dict=True)
        results["models"][name] = {
            "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "test_precision": {k: round(report[k]["precision"], 4) for k in LABELS},
            "test_recall": {k: round(report[k]["recall"], 4) for k in LABELS},
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }

    results["mcnemar_lr_vs_nb"] = mcnemar_test(
        y_test, predictions["logistic_regression"], predictions["naive_bayes"]
    )

    # Leakage check: same model and split, but emoticons left in the text and a
    # whitespace tokenizer (like tweet-aware tokenizers) so ":)" becomes a feature.
    # scikit-learn's default token pattern drops punctuation, which hides the leak.
    leaky = build_models()["logistic_regression"]
    leaky.set_params(tfidf__token_pattern=r"\S+")
    leaky.fit(train["clean_with_emoticons"], y_train)
    leaky_pred = leaky.predict(test["clean_with_emoticons"])
    results["leakage_check"] = {
        "test_accuracy_with_emoticons": round(float(accuracy_score(y_test, leaky_pred)), 4),
        "test_accuracy_without_emoticons": results["models"]["logistic_regression"]["test_accuracy"],
    }

    best_name = max(results["models"], key=lambda n: results["models"][n]["test_macro_f1"])
    best = fitted[best_name]
    results["selected_model"] = best_name
    results["top_features_logistic_regression"] = top_features(fitted["logistic_regression"])

    joblib.dump(best, models_dir / "sentiment.joblib")
    save_confusion_matrix(
        y_test, predictions[best_name], reports_dir / "confusion_matrix.png",
        f"Test set: {best_name.replace('_', ' ')}",
    )
    with open(reports_dir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    results = run(args.models_dir, args.reports_dir)
    print(json.dumps({k: results[k] for k in ("models", "mcnemar_lr_vs_nb", "leakage_check", "selected_model")}, indent=2))


if __name__ == "__main__":
    main()
