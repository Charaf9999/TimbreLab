# TimbreLab - Guide de la partie Basma

## Fichiers

- `models/supervised.py` : préparation, séparation, entraînement et évaluation.
- `models/__init__.py` : import facile depuis l'interface de Chems.
- `tests/test_supervised.py` : tests automatiques sur données inventées.
- `scripts/demo_basma.py` : démonstration sur un CSV audio ou régression.

## Installation

Dans la racine du dépôt partagé `TimbreLab`, créer un environnement Python et installer
`pandas numpy scikit-learn pytest`. **Demander à Chems** d'ajouter ces dépendances au
`requirements.txt` commun ; ne pas écraser celui d'un coéquipier.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy scikit-learn pytest
```

## Tests

```powershell
python -m pytest tests/test_supervised.py -q
```

## Essai avec le CSV GTZAN reçu par le groupe

Dézipper `TimbreLab-dataset-handoff.zip` localement dans `data/processed/`.
Cette destination est ignorée par `.gitignore`. Le CSV réel ne doit pas partir sur Git.

```powershell
python -m scripts.demo_basma data/processed/gtzan_features.csv
```

## Essai avec le CSV de régression déjà disponible dans le dépôt

```powershell
python -m scripts.demo_basma data/examples/regression_fixture.csv
```

## Interface prévue pour Chems

```python
from models import train_supervised
import pandas as pd

table = pd.read_csv("data/processed/gtzan_features.csv")
features = ["tempo_bpm", "rms_mean", "centroid_hz_mean"] + [
    f"mfcc_{i:02d}_mean" for i in range(1, 14)
]
result = train_supervised(
    table, task="classification", target="genre", features=features,
    model_name="svm", group_column="recording_id"
)
print(result["metrics"])
print(result["predictions"].head())
# result["pipeline"] est déjà ajusté et accepte table_de_nouveaux_clips[features].
# result["metadata"] conserve noms des caractéristiques, split et feature_version.
```

## Convention / garde-fous

- `task` : `classification` ou `regression`.
- Classifieurs : `logistic_regression`, `decision_tree`, `gaussian_nb`, `svm`, `knn`, `neural_network`, `random_forest`.
- Régressions : `linear_simple` (une variable), `linear_multiple` (au moins deux), `polynomial` (une variable ; degré 2 par défaut, configurable avec `degree=`).
- Pour `genre` avec la colonne `recording_id`, le découpage se fait automatiquement par groupes.
- Séparation train/test reproductible (`random_state=42`, test 20 % environ) ;
  imputation et standardisation **après** la séparation dans un pipeline.
- Si une classe est absente d'un des deux ensembles, erreur claire : ne jamais mélanger
  les groupes pour contourner le problème.
- La métrique principale de classification est `f1_macro` ; pour la régression, `mae` et `r2`.
- Les scores de comparaison ne constituent **pas** une procédure de choix d'hyperparamètres
  : ne pas sélectionner le meilleur modèle sur le test et ensuite annoncer le même score
  comme estimation finale non biaisée. Prévoir un jeu de validation ou une CV sur train.
- Le schéma audio `audio_v1` comprend exactement les 16 colonnes convenues.
- `random_forest` possède une implémentation provisoire sklearn. Charaf peut brancher
  son implémentation en appelant `register_classifier("random_forest", sa_factory)`.
- L'audit GTZAN décrit des doublons exacts regroupés par `recording_id` ; il signale que
  les doublons approchants / artistes répétés demandent encore vérification.

## Livrables pour Layla

`result["predictions"]` a les colonnes `row_id`, `actual`, `predicted`.
Elle peut afficher une matrice de confusion ou un graphe valeurs réelles/prédites.
Le script de démonstration exporte `data/processed/basma_results/*.csv` (non suivis par Git).
La méthode `compare_models(...)` retourne le tableau de comparaison des scores.

## Rapport

Présenter : données et colonnes utilisées, nettoyage, imputation, séparation par
`recording_id`, modèles testés, réglages, mêmes données test, tableaux Accuracy/F1 macro,
MAE/R², erreurs confondues, comparaison au modèle Dummy et limites. Les exemples CSV
inventés servent uniquement à vérifier le fonctionnement, pas à publier des scores.

## GitHub

La branche personnelle deja creee est `models/basma-supervised`. Consultez `README_BASMA_COMMENCER.md` pour le commit et le push; ne recreez pas une deuxieme branche.

Ouvrir ensuite une Pull Request vers `main` et demander à Layla/Chems de la relire.
