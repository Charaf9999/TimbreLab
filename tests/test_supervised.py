"""Tests rapides, avec données inventées (pas des scores scientifiques)."""
import numpy as np
import pandas as pd
import pytest
from models.supervised import train_supervised, compare_models


def audio_example():
    rng = np.random.default_rng(123)
    genres = ["blues", "jazz"]
    rows = []
    for i in range(50):
        genre = genres[i % 2]
        row = {
            "clip_id": f"clip_{i}",
            "recording_id": f"record_{i}",
            "genre": genre,
            "tempo_bpm": float(100 + 20 * (i % 2) + rng.normal()),
            "rms_mean": float(.1 + rng.normal(scale=.01)),
            "centroid_hz_mean": float(2000 + 40 * (i % 2) + rng.normal()),
        }
        row.update({f"mfcc_{j:02d}_mean": float(rng.normal()) for j in range(1, 14)})
        rows.append(row)
    return pd.DataFrame(rows)


def regression_example():
    rng = np.random.default_rng(1)
    bills = np.linspace(10, 90, 60)
    parties = np.tile([1, 2, 3, 4, 5], 12)
    tips = 2 + .15 * bills + .4 * parties + rng.normal(scale=.1, size=len(bills))
    return pd.DataFrame({"bill": bills, "party_size": parties, "tip": tips})


@pytest.mark.parametrize("name", [
    "logistic_regression", "decision_tree", "gaussian_nb", "svm", "knn", "neural_network", "random_forest"
])
def test_classifier(name):
    df = audio_example()
    feats = [c for c in df.columns if c not in ("clip_id", "recording_id", "genre")]
    result = train_supervised(df, "classification", "genre", feats, name)
    assert set(result) == {"pipeline", "metrics", "predictions", "metadata"}
    assert list(result["predictions"].columns) == ["row_id", "actual", "predicted"]
    assert result["metadata"]["group_column"] == "recording_id"
    assert result["metadata"]["feature_version"] == "audio_v1"
    assert 0 <= result["metrics"]["accuracy"] <= 1


@pytest.mark.parametrize("name,feats", [
    ("linear_simple", ["bill"]),
    ("linear_multiple", ["bill", "party_size"]),
    ("polynomial", ["bill"]),
])
def test_regressors(name, feats):
    result = train_supervised(regression_example(), "regression", "tip", feats, name)
    assert "mae" in result["metrics"]
    assert "r2" in result["metrics"]
    assert len(result["predictions"]) > 0


def test_leakage_rejected():
    df = audio_example()
    with pytest.raises(ValueError, match="identifiants"):
        train_supervised(df, "classification", "genre", ["clip_id"], "svm")


def test_missing_features_and_missing_target():
    df = regression_example()
    df.loc[0, "bill"] = np.nan
    df.loc[1, "tip"] = np.nan
    result = train_supervised(df, "regression", "tip", ["bill"], "linear_simple")
    assert result["metadata"]["dropped_missing_targets"] == 1


def test_reproducible_comparison():
    df = regression_example()
    left = compare_models(df, "regression", "tip", ["bill"], ["linear_simple", "polynomial"])
    right = compare_models(df, "regression", "tip", ["bill"], ["linear_simple", "polynomial"])
    pd.testing.assert_frame_equal(left, right)
