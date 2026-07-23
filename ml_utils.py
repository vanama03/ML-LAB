"""Shared utilities for the ML-LAB notebooks.

This module centralizes code that was previously duplicated across
``INLAB_3.ipynb``, ``POSTLAB_3.ipynb``, ``TASK_1.ipynb`` and ``TASK_2.ipynb``:
dataset loading (Iris, Sonar, Titanic), feature scaling + train/test splitting,
model training/evaluation, and the common matplotlib/seaborn plots.

Import the helpers you need in a notebook, e.g.::

    from ml_utils import load_iris_split, evaluate_model, plot_confusion_matrix
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Student identifier used as a prefix in plot titles across the notebooks.
STUDENT_ID = "24EU02053"

# Public dataset URLs used across the notebooks.
SONAR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

SONAR_FEATURES = [f"F_{i}" for i in range(1, 61)]


def silence_warnings() -> None:
    """Suppress warnings (mirrors ``warnings.filterwarnings('ignore')`` calls)."""
    warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_iris_scaled():
    """Return standardized Iris features and target as ``(X_scaled, y)``."""
    raw = load_iris()
    X_scaled = StandardScaler().fit_transform(raw.data)
    return X_scaled, raw.target


def load_iris_split(test_size=0.20, random_state=42, stratify=True):
    """Load Iris, standardize features and return a stratified train/test split.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    X_scaled, y = load_iris_scaled()
    return train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )


def load_iris_dataframe():
    """Load Iris as a DataFrame with ``species_id`` and ``species_name`` columns."""
    raw = load_iris()
    df = pd.DataFrame(data=raw.data, columns=raw.feature_names)
    df["species_id"] = raw.target
    species_mapping = {i: name for i, name in enumerate(raw.target_names)}
    df["species_name"] = df["species_id"].map(species_mapping)
    return df, list(raw.feature_names)


def load_sonar(scale=True):
    """Load the Sonar dataset.

    Returns ``(df, X, y, X_scaled)`` where ``df`` has the named feature columns
    plus a ``Class`` column and a numeric ``Class_numeric`` column (M -> 1,
    R -> 0). ``X`` is the feature DataFrame, ``y`` the numeric target and
    ``X_scaled`` the standardized features (``None`` when ``scale=False``).
    """
    columns = SONAR_FEATURES + ["Class"]
    df = pd.read_csv(SONAR_URL, header=None, names=columns)
    df["Class_numeric"] = df["Class"].map({"M": 1, "R": 0})

    X = df[SONAR_FEATURES]
    y = df["Class_numeric"]
    X_scaled = StandardScaler().fit_transform(X) if scale else None
    return df, X, y, X_scaled


def load_sonar_split(test_size=0.30, random_state=42, stratify=True):
    """Load Sonar, standardize features and return a stratified train/test split.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    _, _, y, X_scaled = load_sonar(scale=True)
    return train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )


def load_titanic_raw():
    """Load the raw Titanic dataset from the public GitHub mirror."""
    return pd.read_csv(TITANIC_URL)


def preprocess_titanic(df=None, drop_columns=("PassengerId", "Name", "Ticket")):
    """Apply the common Titanic cleaning steps shared by the notebooks.

    Steps: fill ``Age`` with its median, fill ``Embarked`` with its mode, drop
    the sparse ``Cabin`` column and any identifier columns in ``drop_columns``.
    Returns a cleaned copy of the DataFrame.
    """
    if df is None:
        df = load_titanic_raw()
    df = df.copy()

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    existing = [c for c in drop_columns if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model", verbose=True):
    """Fit ``model``, predict on the test set and report accuracy.

    Returns ``(model, predictions, accuracy)``.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    if verbose:
        print(f"{name} Test Accuracy: {accuracy * 100:.2f}%")
    return model, preds, accuracy


def print_classification_report(y_true, y_pred, target_names=None, title="Classification Report"):
    """Print a labeled classification report block."""
    print(f"--- {title} ---")
    print(classification_report(y_true, y_pred, target_names=target_names))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _titled(title):
    """Prefix plot titles with the student id, matching the notebook style."""
    return f"{STUDENT_ID} -> {title}"


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix Heatmap",
                          cmap="Blues", figsize=(5, 4)):
    """Plot a confusion matrix as an annotated seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(_titled(title))
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()
    return cm


def plot_correlation_heatmap(df, title="Correlation Heatmap", annot=True,
                             cmap="coolwarm", figsize=(8, 6)):
    """Plot a correlation heatmap for the numeric columns of ``df``."""
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=annot, cmap=cmap)
    plt.title(_titled(title))
    plt.show()
