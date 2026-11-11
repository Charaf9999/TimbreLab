
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_clustering(table, features, k):
    """Regrouper les morceaux selon leurs caractéristiques numériques."""

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Fournir un tableau pandas non vide.")

    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("Choisir au moins une caractéristique numérique.")

    if len(features) != len(set(features)):
        raise ValueError("Les caractéristiques doivent être uniques.")

    if set(features) & {"clip_id", "recording_id", "genre"}:
        raise ValueError("Les identifiants et le genre sont exclus du clustering.")

    absent = [col for col in features if col not in table.columns]
    if absent:
        raise ValueError(f"Colonnes absentes : {absent}")

    if any(not pd.api.types.is_numeric_dtype(table[col]) for col in features):
        raise ValueError("Toutes les caractéristiques doivent être numériques.")

    if isinstance(k, bool) or not isinstance(k, int) or not (2 <= k < len(table)):
        raise ValueError("k doit être compris entre 2 et le nombre de lignes moins 1.")

    # Préparation des caractéristiques
    X = table.loc[:, features].replace([np.inf, -np.inf], np.nan)

    if X.isna().all().any():
        raise ValueError("Une caractéristique est entièrement manquante.")

    # Prétraitement et clustering dans un même pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
    ])

    labels = pipeline.fit_predict(X)

    # Évaluation du clustering
    n_clusters = len(np.unique(labels))
    X_scaled = pipeline[:-1].transform(X)

    silhouette = (
        float(silhouette_score(X_scaled, labels))
        if 2 <= n_clusters < len(table)
        else None
    )

    # Association de chaque morceau à son cluster
    row_ids = (
        table["clip_id"].to_numpy()
        if "clip_id" in table
        else table.index.to_numpy()
    )

    predictions = pd.DataFrame({
        "row_id": row_ids,
        "cluster": labels,
    })

    return {
        "pipeline": pipeline,
        "metrics": {
            "silhouette_score": silhouette,
            "n_clusters_found": n_clusters,
        },
        "predictions": predictions,
        "metadata": {
            "task": "clustering",
            "model_name": "kmeans",
            "features": list(features),
            "settings": {"k": k, "random_state": 42},
            "k": k,
            "random_state": 42,
        },
    }
