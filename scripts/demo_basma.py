"""Commande: python -m scripts.demo_basma CHEMIN_DU_CSV

CSV audio : colonnes clip_id, recording_id, genre et 16 caractéristiques.
CSV régression : colonnes bill, party_size, tip (exemple du projet).
"""
import sys
import json
from pathlib import Path
import pandas as pd
from models.supervised import AUDIO_FEATURES, compare_models, train_supervised


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.demo_basma CHEMIN_DU_CSV")
        return 1
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Fichier introuvable : {path}")
        return 1
    table = pd.read_csv(path)
    print(f"Fichier : {path} | Dimensions : {table.shape}")
    out = Path("data/processed/basma_results")
    out.mkdir(parents=True, exist_ok=True)
    if "genre" in table:
        feats = list(AUDIO_FEATURES)
        result = train_supervised(table, "classification", "genre", feats, "logistic_regression")
        print("\n=== Exemple classification ===")
        print("Scores :", result["metrics"])
        print(result["predictions"].head().to_string(index=False))
        model_names = [
            "logistic_regression", "decision_tree", "gaussian_nb", "svm",
            "knn", "neural_network", "random_forest"
        ]
        comparison = compare_models(table, "classification", "genre", feats, model_names)
        print("\nComparaison :")
        print(comparison.to_string(index=False))
        comparison.to_csv(out / "classification_comparison.csv", index=False)
        result["predictions"].to_csv(out / "classification_predictions.csv", index=False)
        labels = sorted(table["genre"].dropna().unique())
        matrix = pd.crosstab(
            result["predictions"]["actual"], result["predictions"]["predicted"],
            rownames=["reel"], colnames=["predit"]
        ).reindex(index=labels, columns=labels, fill_value=0)
        matrix.to_csv(out / "confusion_logistic_regression.csv")
        with open(out / "classification_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, ensure_ascii=False, indent=2)
    elif {"bill", "party_size", "tip"}.issubset(table.columns):
        print("\n=== Régression simple/polynomiale ===")
        comparison = compare_models(table, "regression", "tip", ["bill"],
            ["linear_simple", "polynomial"])
        print(comparison.to_string(index=False))
        comparison.to_csv(out / "regression_simple_polynomial.csv", index=False)
        print("\n=== Régression multiple ===")
        result = train_supervised(table, "regression", "tip", ["bill", "party_size"], "linear_multiple")
        print("Scores :", result["metrics"])
        print(result["predictions"].head().to_string(index=False))
        result["predictions"].to_csv(out / "regression_predictions.csv", index=False)
        with open(out / "regression_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, ensure_ascii=False, indent=2)
    else:
        print("CSV non reconnu : il faut le CSV audio standard ou bill/party_size/tip.")
        return 2
    print(f"\nRésultats enregistrés localement : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
