from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def plot_genre_distribution(df):
    """Afficher le nombre de morceaux par genre musical."""
    counts = df["genre"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Répartition des genres musicaux")
    ax.set_xlabel("Genre musical")
    ax.set_ylabel("Nombre de morceaux")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def plot_missing_values(df):
    """Visualiser les valeurs manquantes du dataset."""
    missing = df.isna().sum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(missing.index, missing.values)
    ax.set_title("Valeurs manquantes par colonne")
    ax.set_xlabel("Caractéristiques")
    ax.set_ylabel("Nombre de valeurs manquantes")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()

    return fig


def plot_feature_distribution(df, feature, title, unit):
    """Afficher la distribution d'une caractéristique numérique."""
    if feature not in df.columns:
        raise ValueError(f"Colonne introuvable : {feature}")

    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    if values.empty:
        raise ValueError("Aucune valeur numérique exploitable.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=30, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(unit)
    ax.set_ylabel("Nombre de morceaux")
    fig.tight_layout()

    return fig

def plot_clusters_pca(table, result):
    """Visualiser les clusters avec une projection PCA en 2D."""

    features = result["metadata"]["features"]
    pipeline = result["pipeline"]
    labels = result["predictions"]["cluster"].to_numpy()

    X = table[features].replace([np.inf, -np.inf], np.nan)
    X_scaled = pipeline[:-1].transform(X)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster

        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=f"Cluster {cluster}",
            alpha=0.65,
            s=25
        )

    ax.set_title("Visualisation des clusters K-means (PCA)")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    return fig


def plot_cluster_profiles(table, result):
    """Comparer les caractéristiques moyennes standardisées."""

    features = result["metadata"]["features"]
    labels = result["predictions"]["cluster"].to_numpy()

    X = table[features].replace([np.inf, -np.inf], np.nan)
    X_scaled = result["pipeline"][:-1].transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df["cluster"] = labels

    selected = [
        "tempo_bpm",
        "rms_mean",
        "centroid_hz_mean"
    ]

    means = scaled_df.groupby("cluster")[selected].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        means.values,
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        aspect="auto"
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Tempo", "Énergie", "Brillance"])
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(
        [f"Cluster {i}" for i in means.index]
    )

    ax.set_title("Profils acoustiques des clusters")
    fig.colorbar(
        image, ax=ax,
        label="Moyenne standardisée"
    )

    fig.tight_layout()
    return fig

def cluster_genre_table(table, result):
    """Comparer les clusters aux vrais genres musicaux."""

    if "genre" not in table.columns:
        raise ValueError("La colonne genre est absente.")

    labels = result["predictions"]["cluster"].to_numpy()

    if len(labels) != len(table):
        raise ValueError("Nombre de clusters et de lignes incompatible.")

    comparison = pd.DataFrame({
        "cluster": labels,
        "genre": table["genre"].to_numpy()
    })

    counts = pd.crosstab(
        comparison["cluster"],
        comparison["genre"]
    )

    return counts


def plot_cluster_genres(table, result):
    """Afficher la distribution des genres par cluster."""

    counts = cluster_genre_table(table, result)

    percentages = counts.div(
        counts.sum(axis=1), axis=0
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    percentages.plot(
        kind="bar",
        stacked=True,
        colormap="tab20",
        ax=ax
    )

    ax.set_title(
        "Répartition des genres musicaux par cluster"
    )
    ax.set_xlabel("Cluster K-means")
    ax.set_ylabel("Pourcentage des morceaux (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)

    ax.legend(
        title="Genre musical",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()
    return fig

def plot_confusion_matrix(predictions):
    """Créer une matrice de confusion pour la classification."""

    required = {"actual", "predicted"}

    if not required.issubset(predictions.columns):
        raise ValueError("Colonnes actual et predicted requises.")

    if predictions.empty:
        raise ValueError("Le tableau des prédictions est vide.")

    if predictions[["actual", "predicted"]].isna().any().any():
        raise ValueError("Les étiquettes ne doivent pas être manquantes.")

    y_true = predictions["actual"].astype(str)
    y_pred = predictions["predicted"].astype(str)

    labels = sorted(set(y_true) | set(y_pred))

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False
    )

    ax.set_title("Matrice de confusion - Classification")
    ax.set_xlabel("Genre prédit")
    ax.set_ylabel("Genre réel")

    fig.tight_layout()
    return fig


def plot_regression_predictions(predictions):
    """Comparer les valeurs réelles et prédites en régression."""

    required = {"actual", "predicted"}

    if not required.issubset(predictions.columns):
        raise ValueError("Colonnes actual et predicted requises.")

    if predictions.empty:
        raise ValueError("Le tableau des prédictions est vide.")

    actual = pd.to_numeric(predictions["actual"], errors="coerce")
    predicted = pd.to_numeric(predictions["predicted"], errors="coerce")

    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError("Les valeurs réelles doivent être numériques.")

    if not np.isfinite(predicted.to_numpy()).all():
        raise ValueError("Les prédictions doivent être numériques.")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(actual, predicted, alpha=0.7)

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Prédiction parfaite"
    )

    ax.set_title("Régression : valeurs réelles vs prédites")
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Valeurs prédites")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
