"""Préparation, entraînement et évaluation des modèles supervisés de TimbreLab.

Le prétraitement est ajusté uniquement sur les données d'entraînement.
Pour GTZAN, recording_id protège contre les doublons entre train et test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
AUDIO_FEATURES = (
    "tempo_bpm", "rms_mean", "centroid_hz_mean",
    *(f"mfcc_{i:02d}_mean" for i in range(1, 14)),
)

# Charaf peut enregistrer ici sa factory Random Forest avec register_classifier().
CLASSIFIERS: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "gaussian_nb": lambda: GaussianNB(),
    "svm": lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "neural_network": lambda: MLPClassifier(
        hidden_layer_sizes=(32,), max_iter=1000, random_state=RANDOM_STATE
    ),
    # Estimateur temporaire utilisable en attendant la factory de Charaf.
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
}
REGRESSORS = {"linear_simple", "linear_multiple", "polynomial"}


def register_classifier(name: str, factory: Callable[[], BaseEstimator]) -> None:
    """Enregistrer un classifieur externe, notamment le Random Forest de Charaf."""
    if not isinstance(name, str) or not name:
        raise ValueError("Le nom du classifieur ne doit pas être vide.")
    if not callable(factory):
        raise ValueError("factory doit être une fonction qui retourne un estimateur.")
    CLASSIFIERS[name] = factory


def _validated_data(table, task, target, features, group_column, model_name, degree):
    if not isinstance(table, pd.DataFrame):
        raise ValueError("table doit être un DataFrame pandas.")
    if task not in ("classification", "regression"):
        raise ValueError("task doit être 'classification' ou 'regression'.")
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("Choisissez au moins une caractéristique numérique.")
    features = list(features)
    if len(features) != len(set(features)):
        raise ValueError("La liste des caractéristiques contient des doublons.")
    if target not in table.columns:
        raise ValueError(f"Colonne cible introuvable : {target}.")
    if any(f not in table.columns for f in features):
        raise ValueError("Caractéristiques introuvables : " + ", ".join(
            str(f) for f in features if f not in table.columns
        ))
    if target in features or any(f in ("clip_id", "recording_id", "row_id") for f in features):
        raise ValueError("La cible et les identifiants ne doivent pas être des prédicteurs.")
    if not all(pd.api.types.is_numeric_dtype(table[f]) for f in features):
        raise ValueError("La première version accepte uniquement des caractéristiques numériques.")
    if task == "classification" and model_name not in CLASSIFIERS:
        raise ValueError(f"Modèle de classification inconnu : {model_name}.")
    if task == "regression" and model_name not in REGRESSORS:
        raise ValueError(f"Modèle de régression inconnu : {model_name}.")
    if model_name == "linear_simple" and len(features) != 1:
        raise ValueError("La régression linéaire simple nécessite exactement 1 caractéristique.")
    if model_name == "linear_multiple" and len(features) < 2:
        raise ValueError("La régression multiple nécessite au moins 2 caractéristiques.")
    if model_name == "polynomial":
        if len(features) != 1:
            raise ValueError("La régression polynomiale nécessite exactement 1 caractéristique.")
        if not isinstance(degree, int) or not (2 <= degree <= 5):
            raise ValueError("Le degré polynomial doit être un entier de 2 à 5.")

    if group_column is None and target == "genre" and "recording_id" in table.columns:
        group_column = "recording_id"
    if group_column is not None and group_column not in table.columns:
        raise ValueError(f"Colonne de regroupement introuvable : {group_column}.")

    # Conserver les indices originaux pour retrouver la ligne d'où vient le résultat.
    work = table.copy()
    work["__row_id__"] = table["clip_id"].to_numpy() if "clip_id" in table else table.index.to_numpy()
    count_before = len(work)
    work = work.dropna(subset=[target]).reset_index(drop=True)
    if work.empty or len(work) < 6:
        raise ValueError("Données insuffisantes après suppression des cibles manquantes (6 lignes minimum).")
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(work[target]):
            raise ValueError("La cible de régression doit être numérique.")
        if not np.isfinite(work[target].astype(float)).all():
            raise ValueError("La cible de régression contient des valeurs infinies ou invalides.")
    elif work[target].nunique() < 2:
        raise ValueError("La classification nécessite au moins deux classes différentes.")
    if task == "classification" and work[target].value_counts().min() < 2:
        raise ValueError("Chaque classe doit contenir au moins deux exemples.")
    if group_column is not None:
        if work[group_column].isna().any():
            raise ValueError("Les identifiants de groupe doivent être présents sur toutes les lignes.")
        if work[group_column].nunique() < 2:
            raise ValueError("Il faut au moins deux groupes distincts.")
    # Les valeurs manquantes peuvent être imputées ; les colonnes totalement vides non.
    X = work[features].replace([np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        bad = X.columns[X.isna().all()].tolist()
        raise ValueError("Caractéristiques complètement vides : " + ", ".join(bad))
    return work, X, group_column, count_before - len(work)


def train_supervised(
    table: pd.DataFrame,
    task: str,
    target: str,
    features: list[str],
    model_name: str,
    group_column: str | None = None,
    degree: int = 2,
) -> dict:
    """Retourne un dict avec pipeline, metrics, predictions, metadata.

    task: 'classification' ou 'regression'.
    model_name (classification): logistic_regression, decision_tree, gaussian_nb,
        svm, knn, neural_network, random_forest.
    model_name (regression): linear_simple, linear_multiple, polynomial.
    degree: degré de la régression polynomiale (2 par défaut).
    """
    work, X, group_column, dropped_targets = _validated_data(
        table, task, target, features, group_column, model_name, degree
    )
    y = work[target]
    if group_column is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=work[group_column]))
    else:
        indices = np.arange(len(work))
        stratify = y if task == "classification" else None
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
            )
        except ValueError as exc:
            raise ValueError(f"Découpage impossible avec ces classes : {exc}") from exc

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if len(X_test) < 2:
        raise ValueError("Le jeu de test contient moins de deux lignes : ajoutez des données.")
    if X_train.isna().all().any():
        raise ValueError("Au moins une caractéristique est entièrement manquante dans le train.")
    if task == "classification":
        expected_classes = set(y.unique())
        if set(y_train.unique()) != expected_classes or set(y_test.unique()) != expected_classes:
            raise ValueError(
                "La séparation train/test ne contient pas toutes les classes des deux côtés. "
                "Ajoutez des exemples/groupes ou adaptez le découpage sans mélanger les enregistrements."
            )
        if model_name == "knn" and len(X_train) < 5:
            raise ValueError("KNN nécessite au moins cinq exemples d'entraînement.")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "polynomial":
        steps.append(("polynomial", PolynomialFeatures(degree=degree, include_bias=False)))
    if model_name not in ("decision_tree", "random_forest"):
        steps.append(("scaler", StandardScaler()))
    preprocessing = Pipeline(steps)

    if task == "classification":
        estimator = CLASSIFIERS[model_name]()
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        estimator = LinearRegression()
        baseline = DummyRegressor(strategy="mean")
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
            "baseline_f1_macro": float(f1_score(y_test, baseline_pred, average="macro", zero_division=0)),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "r2": float(r2_score(y_test, predicted)),
            "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
            "baseline_r2": float(r2_score(y_test, baseline_pred)),
        }
    preds = pd.DataFrame({
        "row_id": work.iloc[test_idx]["__row_id__"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predicted,
    })
    metadata = {
        "task": task,
        "target": target,
        "model_name": model_name,
        "features": list(features),
        "group_column": group_column,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": len(train_idx),
        "test_rows": len(test_idx),
        "dropped_missing_targets": dropped_targets,
        "polynomial_degree": degree if model_name == "polynomial" else None,
        "feature_version": "audio_v1" if target == "genre" and tuple(features) == AUDIO_FEATURES else None,
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": preds,
        "metadata": metadata,
    }


def compare_models(
    table: pd.DataFrame,
    task: str,
    target: str,
    features: list[str],
    model_names: list[str],
    group_column: str | None = None,
    degree: int = 2,
) -> pd.DataFrame:
    """Compare plusieurs modèles sur un découpage reproductible identique.

    Utiliser des lignes et colonnes identiques pour chaque modèle ; ne pas optimiser
    les hyperparamètres à partir des résultats du test.
    """
    rows = []
    for name in model_names:
        result = train_supervised(table, task, target, features, name, group_column, degree)
        rows.append({"model_name": name, **result["metrics"]})
    return pd.DataFrame(rows)
