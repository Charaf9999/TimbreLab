# TimbreLab - Ma partie Basma (MODE D'EMPLOI)

Ce ZIP contient **uniquement la partie Basma**, avec 2 petits CSV de démonstration inventés. Il ne contient pas l'application de Chems, le code de Charaf ni les vraies musiques GTZAN.

## 1. Où placer les fichiers ?

Vous avez DEJA clone `TimbreLab` sur le Bureau et cree la branche `models/basma-supervised`.

1. Ouvrez `C:\Users\HP\Desktop\TimbreLab` dans VS Code.
2. Verifiez votre branche dans le terminal :

```powershell
git branch
```

L'etoile doit etre devant `models/basma-supervised`.

3. Telechargez ce ZIP dans **Telechargements**.
4. Sans creer un deuxieme dossier `TimbreLab`, depuis le terminal place a la racine du projet, executez :

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\TimbreLab_Basma_Pret_a_Copier_v2.zip" -DestinationPath .
```

Si PowerShell vous signale qu'un fichier existe deja, arretez et comparez avant de le remplacer. Il ne faut pas ecraser le travail d'un autre membre.

Vous devez avoir `models/supervised.py`, `tests/test_supervised.py`, `scripts/demo_basma.py`, `docs/basma-guide.md`, et 2 exemples `data/examples/basma_*.csv`.

## 2. Installer les dependances Python

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r basma-requirements.txt
```

On utilise directement `\.venv\Scripts\python.exe` pour eviter les problemes d'activation PowerShell.

## 3. Executer les tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_supervised.py -q
```

Resultat attendu : 13 tests reussis.

## 4. Tester avec les deux CSV INCLUS dans ce ZIP

Classification :

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_basma data/examples/basma_audio_demo.csv
```

Regression :

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_basma data/examples/basma_regression_demo.csv
```

Les résultats apparaissent dans `data/processed/basma_results/` (sortie locale).
ATTENTION : les valeurs des petits CSV sont **inventees** pour tester le code : ne pas utiliser les scores pour le rapport scientifique.

## 5. Tester avec les vraies donnees de Charaf (apres validation)

Decompresser `TimbreLab-dataset-handoff.zip` dans `data/processed/` de votre depot local, puis :

```powershell
.\.venv\Scripts\python.exe -m scripts.demo_basma data/processed/gtzan_features.csv
```

Ne pas pousser GTZAN, `.venv`, les modèles entraines ou les resultats temporaires sur GitHub. La detection des doublons exacts repose sur `recording_id`; completer l'audit des quasi-doublons avant l'evaluation finale.

## 6. Envoyer sur GitHub

```powershell
git status
git add models/ tests/test_supervised.py scripts/ docs/basma-guide.md data/examples/basma_audio_demo.csv data/examples/basma_regression_demo.csv basma-requirements.txt README_BASMA_COMMENCER.md
git diff --cached --name-only
git commit -m "Add Basma supervised ML models and tests"
git push -u origin models/basma-supervised
```

Ouvrir une Pull Request de `models/basma-supervised` vers `main`.

## 7. Utilisation par Chems

```python
from models import train_supervised
result = train_supervised(table, "classification", "genre", features, "svm", group_column="recording_id")
# result contient 'pipeline', 'metrics', 'predictions', 'metadata'
```

Layla peut creer ses figures a partir des colonnes `row_id`, `actual`, `predicted` du tableau de resultats.

**Ce qui reste pour la version finale** : validation sur les donnees reelles, choix des modeles sans utiliser le test pour regler les hyperparametres, integration du Random Forest de Charaf, documentation et section du rapport. Voir `docs/basma-guide.md`.
