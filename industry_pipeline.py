"""ML pipeline example for iris dataset.

This script demonstrates an industry-style data science workflow using scikit-learn.
It loads the Iris dataset, splits data, builds a pipeline with preprocessing and
modeling steps, evaluates the model, and saves the trained model to disk.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("iris_model.joblib")


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the iris dataset and return features and target.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features ``X`` and target ``y`` as pandas objects.
    """
    dataset = load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


def build_pipeline() -> Pipeline:
    """Create a scikit-learn pipeline with preprocessing and model.

    Returns
    -------
    Pipeline
        A pipeline performing feature scaling followed by logistic regression.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, n_jobs=1))
    ])


def train_and_evaluate() -> Tuple[Pipeline, str]:
    """Train the model and generate a classification report.

    Returns
    -------
    Tuple[Pipeline, str]
        The fitted pipeline and text classification report.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    report = classification_report(y_test, predictions)
    return pipe, report


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Persist the trained model to disk."""
    joblib.dump(model, path)


if __name__ == "__main__":
    model, report = train_and_evaluate()
    print(report)
    save_model(model)
