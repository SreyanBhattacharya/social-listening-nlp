"""Social listening on 20,000 tweets about the 2015 UK general election.

Run (after training):  python -m sociallistening.topics

What it does
1. Keeps English tweets and cleans them (URLs, mentions and the "RT" prefix removed).
2. Discovers topics with TF-IDF + NMF (non-negative matrix factorisation) on the
   unique tweet texts, so heavily retweeted messages do not dominate the topics.
3. Assigns every tweet (retweets included) to its strongest topic and tracks topic
   volume in 5-minute windows to show what is trending.
4. Counts the top hashtags.
5. Scores each topic's tone with the trained sentiment model. The model was trained
   on general tweets, not politics, so treat these shares as indicative only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .data import load_stream
from .preprocess import clean_text, extract_hashtags

SEED = 42
EXTRA_STOP_WORDS = {"amp", "just", "says", "said", "like", "via", "don", "don't", "it's", "i'm"}


def discover_topics(texts: pd.Series, n_topics: int, n_terms: int = 8):
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS),
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 2),
    )
    unique = texts.drop_duplicates()
    matrix = vectorizer.fit_transform(unique)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=SEED, max_iter=400)
    nmf.fit(matrix)
    vocab = vectorizer.get_feature_names_out()
    topics = []
    for idx, weights in enumerate(nmf.components_):
        terms = [vocab[i] for i in weights.argsort()[::-1][:n_terms]]
        topics.append({"id": idx, "label": " / ".join(terms[:3]), "top_terms": terms})
    return vectorizer, nmf, topics


def run(models_dir: Path, reports_dir: Path, n_topics: int) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = load_stream()
    df = df[df["lang"] == "en"].copy()
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)

    vectorizer, nmf, topics = discover_topics(df["clean"], n_topics)
    weights = nmf.transform(vectorizer.transform(df["clean"]))
    df["topic"] = weights.argmax(axis=1)
    df.loc[weights.max(axis=1) == 0, "topic"] = -1  # no topic terms present

    model_path = models_dir / "sentiment.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        df["p_positive"] = model.predict_proba(df["clean"])[:, 1]
    else:
        df["p_positive"] = np.nan

    assigned = df[df["topic"] >= 0]
    for t in topics:
        rows = assigned[assigned["topic"] == t["id"]]
        t["tweets"] = int(len(rows))
        t["share_of_tweets"] = round(len(rows) / len(df), 3)
        t["retweet_share"] = round(float(rows["is_retweet"].mean()), 3)
        t["predicted_positive_share"] = (
            round(float((rows["p_positive"] > 0.5).mean()), 3) if model_path.exists() else None
        )
    topics.sort(key=lambda t: t["tweets"], reverse=True)

    hashtags = Counter(tag for text in df["text"] for tag in extract_hashtags(text))

    df["window"] = df["created_at"].dt.floor("5min")
    trend = assigned.assign(window=df.loc[assigned.index, "window"]).pivot_table(
        index="window", columns="topic", values="text", aggfunc="count", fill_value=0
    )
    # The first and last windows are only partly covered by the collection period.
    trend = trend.iloc[1:-1]

    summary = {
        "tweets_analysed": int(len(df)),
        "unassigned_tweets": int((df["topic"] < 0).sum()),
        "time_range_utc": [str(df["created_at"].min()), str(df["created_at"].max())],
        "retweet_share": round(float(df["is_retweet"].mean()), 3),
        "topics": topics,
        "top_hashtags": hashtags.most_common(15),
    }
    with open(reports_dir / "topics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot_topics(topics, reports_dir / "topics.png")
    _plot_trend(trend, topics, reports_dir / "topic_trend.png")
    return summary


def _plot_topics(topics: list[dict], path: Path) -> None:
    labels = [t["label"] for t in topics][::-1]
    counts = [t["tweets"] for t in topics][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(topics) + 1.2))
    ax.barh(labels, counts, color="#2f5fb3")
    ax.set_xlabel("Tweets")
    ax.set_title("Topics discovered with NMF (top 3 terms each)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_trend(trend: pd.DataFrame, topics: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for t in topics[:5]:
        if t["id"] in trend.columns:
            ax.plot(trend.index, trend[t["id"]], marker="o", markersize=3, label=t["label"])
    ax.set_ylabel("Tweets per 5 minutes")
    ax.set_xlabel("Time (UTC), 30 April 2015")
    ax.set_title("Topic volume over time (top 5 topics)", fontsize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--topics", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.models_dir, args.reports_dir, args.topics)
    for t in summary["topics"]:
        print(f"{t['tweets']:>6}  {t['label']}")
    print("Top hashtags:", summary["top_hashtags"][:8])


if __name__ == "__main__":
    main()
