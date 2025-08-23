"""Comcast complaint classification pipeline.

This script builds a text classification model that labels Comcast
customer complaints into broad categories (e.g., billing or internet).
It uses a TfidfVectorizer combined with logistic regression, evaluates
on a holdout set, and persists the trained model to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("comcast.csv")
MODEL_PATH = Path("comcast_model.joblib")


def load_data(path: Path = DATA_PATH) -> Tuple[pd.Series, pd.Series]:
    """Load complaint texts and labels from the dataset."""
    df = pd.read_csv(path)
    return df["complaint"], df["category"]


def build_pipeline() -> Pipeline:
    """Create a text-processing and classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=100))
    ])


def train_and_evaluate() -> Tuple[Pipeline, str]:
    """Train the classifier and return the model and report."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds)
    return pipe, report


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Persist the trained model to disk."""
    joblib.dump(model, path)


if __name__ == "__main__":
    model, report = train_and_evaluate()
    print(report)
    save_model(model)
