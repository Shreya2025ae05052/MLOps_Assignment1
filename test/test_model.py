"""Tests for model training, evaluation, saving and inference."""
import joblib


def test_model_predicts_0_or_1(trained_model, clean_df):
    X = clean_df.drop(columns=["target"])
    predictions = trained_model.predict(X)
    assert set(predictions) <= {0, 1}


def test_probabilities_are_valid(trained_model, clean_df):
    X = clean_df.drop(columns=["target"])
    probabilities = trained_model.predict_proba(X)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert abs(probabilities.sum(axis=1) - 1).max() < 1e-6


def test_evaluate_returns_all_metrics(nb, trained_model, clean_df):
    X = clean_df.drop(columns=["target"])
    y = clean_df["target"]
    scores = nb["evaluate"](trained_model, X, y)

    for name in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]:
        assert name in scores
        assert 0 <= scores[name] <= 1


def test_saved_model_gives_same_predictions(trained_model, clean_df, tmp_path):
    X = clean_df.drop(columns=["target"])
    path = tmp_path / "model.joblib"
    joblib.dump(trained_model, path)
    loaded = joblib.load(path)
    assert (loaded.predict(X) == trained_model.predict(X)).all()


def test_predict_function(nb, trained_model, clean_df, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_model, path)
    patient = clean_df.drop(columns=["target"]).iloc[0].to_dict()

    result = nb["predict"](patient, model_path=path)
    assert result["prediction"] in [0, 1]
    assert 0.5 <= result["confidence"] <= 1
