"""Tests for data cleaning and preprocessing."""
import numpy as np


def test_clean_removes_missing_values(nb, raw_df):
    clean = nb["clean_data"](raw_df)
    assert clean.isna().sum().sum() == 0


def test_clean_removes_duplicates(nb, raw_df):
    clean = nb["clean_data"](raw_df)
    assert clean.duplicated().sum() == 0


def test_target_is_binary(nb, raw_df):
    clean = nb["clean_data"](raw_df)
    assert "num" not in clean.columns
    assert set(clean["target"].unique()) <= {0}


def test_all_columns_are_numeric(nb, raw_df):
    clean = nb["clean_data"](raw_df)
    for col in nb["NUMERIC_COLS"] + nb["CATEGORICAL_COLS"]:
        assert np.issubdtype(clean[col].dtype, np.number)


def test_preprocessor_output(nb, clean_df):
    X = clean_df.drop(columns=["target"])
    X_processed = nb["build_preprocessor"]().fit_transform(X)

    assert X_processed.shape[0] == len(X)  # same number of rows
    assert X_processed.shape[1] > len(nb["NUMERIC_COLS"])  # one-hot adds columns
    assert not np.isnan(X_processed).any()
