"""Tests for the evaluation helpers, the insight prompt and the REST API.

A tiny model is trained on toy sentences so the tests run in CI without downloading
the tweet corpus.
"""

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sociallistening import api
from sociallistening.insights import build_prompt, template_brief
from sociallistening.train import build_models, mcnemar_test

TOY_TEXTS = [
    "love this so much", "great happy day", "thanks you are awesome", "what a nice morning",
    "this is terrible", "so sad and tired", "i hate waiting", "awful weather again",
] * 3
TOY_LABELS = [1, 1, 1, 1, 0, 0, 0, 0] * 3


@pytest.fixture()
def client(tmp_path, monkeypatch):
    model = build_models()["logistic_regression"]
    model.set_params(tfidf__min_df=1)
    model.fit(TOY_TEXTS, TOY_LABELS)
    path = tmp_path / "sentiment.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    api.get_model.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()


def test_mcnemar_counts_disagreements():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 1, 0, 1, 0])  # 3 right
    b = np.array([1, 0, 0, 0, 0])  # 3 right
    result = mcnemar_test(y, a, b)
    assert result["a_right_b_wrong"] == 1
    assert result["a_wrong_b_right"] == 1
    assert result["p_value"] == pytest.approx(1.0)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_one_result_per_text(client):
    response = client.post("/predict", json={"texts": ["love this, awesome!", "awful and sad :("]})
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert [p["sentiment"] for p in preds] == ["positive", "negative"]
    assert all(0.0 <= p["positive_probability"] <= 1.0 for p in preds)


def test_predict_rejects_empty_list(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422


SUMMARY = {
    "tweets_analysed": 100,
    "time_range_utc": ["2015-04-30 21:34", "2015-04-30 22:44"],
    "retweet_share": 0.6,
    "top_hashtags": [["bbcqt", 40], ["ge2015", 10]],
    "topics": [
        {"label": "a / b / c", "top_terms": ["a", "b", "c"], "tweets": 60, "share_of_tweets": 0.6,
         "retweet_share": 0.5, "predicted_positive_share": 0.4},
        {"label": "d / e / f", "top_terms": ["d", "e", "f"], "tweets": 40, "share_of_tweets": 0.4,
         "retweet_share": 0.7, "predicted_positive_share": 0.6},
    ],
}


def test_prompt_contains_data_and_guardrails():
    prompt = build_prompt(SUMMARY)
    assert "a / b / c" in prompt and "bbcqt" in prompt
    assert "Use only facts and numbers present in the JSON" in prompt


def test_template_brief_mentions_largest_topic():
    assert "a / b / c" in template_brief(SUMMARY)
