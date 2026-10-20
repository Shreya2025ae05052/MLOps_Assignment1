"""Shared test setup.

The tests use the functions defined in heart_disease_pipeline.ipynb.
Only the notebook cells tagged "functions" are run, so no data is downloaded
and no model is trained. The tests use small fake data instead.
"""
import os

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "heart_disease_pipeline.ipynb")


def load_notebook_functions():
    """Runs the notebook cells tagged "functions" and returns what they define."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "functions" in cell.metadata.get("tags", []):
            exec(cell.source, namespace)
    return namespace


def make_raw_data(n_rows=40):
    """Fake data in the same format as the raw UCI file."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n_rows).astype(float),
        "sex": rng.integers(0, 2, n_rows).astype(float),
        "cp": rng.integers(1, 5, n_rows).astype(float),
        "trestbps": rng.integers(95, 190, n_rows).astype(float),
        "chol": rng.integers(130, 400, n_rows).astype(float),
        "fbs": rng.integers(0, 2, n_rows).astype(float),
        "restecg": rng.integers(0, 3, n_rows).astype(float),
        "thalach": rng.integers(90, 200, n_rows).astype(float),
        "exang": rng.integers(0, 2, n_rows).astype(float),
        "oldpeak": rng.uniform(0, 5, n_rows).round(1),
        "slope": rng.integers(1, 4, n_rows).astype(float),
        "ca": rng.integers(0, 4, n_rows).astype(float).astype(object),
        "thal": rng.choice([3.0, 6.0, 7.0], n_rows).astype(object),
        "num": np.tile([0, 1, 2, 0, 3, 0, 4, 0], n_rows // 8 + 1)[:n_rows],
    })
    # Missing values written as "?", like in the real file
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    # One duplicate row
    df.loc[n_rows - 1] = df.loc[n_rows - 2]
    return df


@pytest.fixture(scope="session")
def nb():
    """Functions and settings from the notebook, e.g. nb["clean_data"]."""
    return load_notebook_functions()


@pytest.fixture
def raw_df():
    return make_raw_data()


@pytest.fixture
def clean_df(nb):
    return nb["clean_data"](make_raw_data())


@pytest.fixture
def trained_model(nb, clean_df):
    """A small Logistic Regression pipeline trained on the fake data."""
    X = clean_df.drop(columns=["target"])
    y = clean_df["target"]
    return nb["build_pipeline"](LogisticRegression(max_iter=1000)).fit(X, y)
