"""Small shared contracts. Changes here should be agreed with all four owners."""

from collections.abc import Mapping

import pandas as pd

from audio.schema import FEATURE_COLUMNS as AUDIO_FEATURES, FEATURE_VERSION as AUDIO_FEATURE_VERSION

MODELS = {
    "classification": {
        "Logistic regression": "logistic_regression",
        "Decision tree": "decision_tree",
        "Gaussian Naive Bayes": "gaussian_naive_bayes",
        "Support vector machine": "svm",
        "K-nearest neighbours": "knn",
        "Random Forest": "random_forest",
        "Neural network (MLP)": "mlp",
    },
    "regression": {
        "Simple linear regression": "simple_linear",
        "Multiple linear regression": "multiple_linear",
        "Polynomial regression": "polynomial",
    },
    "clustering": {"K-means": "kmeans"},
}


def validate_result(result, task, features, model_name=None):
    """Reject incomplete integrations before displaying or exporting a run."""
    if not isinstance(result, Mapping):
        raise ValueError("The analysis module must return a dictionary.")
    required = {"pipeline", "metrics", "predictions", "metadata"}
    if not required.issubset(result):
        raise ValueError("The analysis result is missing pipeline, metrics, predictions or metadata.")
    predictions, metadata = result["predictions"], result["metadata"]
    columns = {"row_id", "cluster"} if task == "clustering" else {"row_id", "actual", "predicted"}
    if not isinstance(predictions, pd.DataFrame) or not columns.issubset(predictions.columns):
        raise ValueError(f"The prediction table needs these columns: {', '.join(sorted(columns))}.")
    if predictions.empty or predictions["row_id"].isna().any() or predictions["row_id"].duplicated().any():
        raise ValueError("Prediction row IDs must be present and unique, with at least one result row.")
    if not isinstance(result["metrics"], Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Metrics and metadata must be dictionaries.")
    if metadata.get("task") != task or list(metadata.get("features", [])) != list(features):
        raise ValueError("The result metadata must match the selected task and ordered features.")
    if not metadata.get("model_name") or not isinstance(metadata.get("settings"), Mapping):
        raise ValueError("Result metadata needs model_name and a settings dictionary.")
    if model_name is not None and metadata["model_name"] != model_name:
        raise ValueError("The returned model does not match the selected model.")
    if not callable(getattr(result["pipeline"], "predict", None)):
        raise ValueError("The returned pipeline must support predict().")
