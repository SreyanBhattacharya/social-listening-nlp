"""Load the NLTK `twitter_samples` corpus into pandas DataFrames.

The corpus has three files:
- positive_tweets.json / negative_tweets.json: 5,000 tweets each, labelled by the
  emoticon that was used to collect them (e.g. ":)" or ":(").
- tweets.20150430-223406.json: 20,000 unlabelled tweets about the 2015 UK general
  election, collected over roughly 70 minutes.

Only the fields needed for analysis are kept (text, timestamp, language). User
profiles and other personal fields are dropped on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

CORPUS = "twitter_samples"
POSITIVE_FILE = "positive_tweets.json"
NEGATIVE_FILE = "negative_tweets.json"
STREAM_FILE = "tweets.20150430-223406.json"


def corpus_dir() -> Path:
    """Return the folder holding the corpus, downloading it with NLTK if needed.

    Set the TWITTER_SAMPLES_DIR environment variable to use a copy you already have.
    """
    override = os.environ.get("TWITTER_SAMPLES_DIR")
    if override:
        return Path(override)

    import nltk

    try:
        return Path(nltk.data.find(f"corpora/{CORPUS}"))
    except LookupError:
        nltk.download(CORPUS, quiet=True)
        return Path(nltk.data.find(f"corpora/{CORPUS}"))


def _read_json_lines(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_labelled(folder: Path | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns `text` and `label` (1 = positive, 0 = negative)."""
    folder = folder or corpus_dir()
    frames = []
    for filename, label in ((POSITIVE_FILE, 1), (NEGATIVE_FILE, 0)):
        rows = _read_json_lines(folder / filename)
        frames.append(pd.DataFrame({"text": [r["text"] for r in rows], "label": label}))
    return pd.concat(frames, ignore_index=True)


def load_stream(folder: Path | None = None) -> pd.DataFrame:
    """Return the unlabelled election tweets with `text`, `created_at`, `lang`, `is_retweet`."""
    folder = folder or corpus_dir()
    rows = _read_json_lines(folder / STREAM_FILE)
    df = pd.DataFrame(
        {
            "text": [r["text"] for r in rows],
            "created_at": pd.to_datetime(
                [r["created_at"] for r in rows], format="%a %b %d %H:%M:%S %z %Y"
            ),
            "lang": [r.get("lang") for r in rows],
        }
    )
    df["is_retweet"] = df["text"].str.startswith("RT ")
    return df
