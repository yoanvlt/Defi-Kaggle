# Défi Kaggle: House Prices - Advanced Regression Techniques

Ce dépôt contient le code pour le défi Kaggle "House Prices". L'objectif est de prédire le prix de vente final (`SalePrice`) de chaque maison.

## Structure du Projet

```
.
├── Data/                   # Données brutes (train.csv, test.csv, etc.)
├── experiments/            # Journal des expériences (experiments.md)
├── notebooks/              # Notebooks Jupyter pour l'analyse et les modèles
├── reports/                # Rapports de synthèse (report.md)
├── src/                    # Scripts Python reproductibles
│   └── run_baseline.py     # Script de la baseline (HistGradientBoosting)
├── submissions/            # Fichiers de soumission au format Kaggle
└── requirements.txt        # Dépendances Python
```

## Installation

1.  **Cloner le dépôt** (si applicable) ou télécharger les fichiers.
2.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Reproduire la Baseline (Run 001)

Pour entraîner le modèle baseline (HistGradientBoostingRegressor), évaluer ses performances par validation croisée et générer le fichier de soumission, exécutez :

```bash
python src/run_baseline.py
```

Les résultats (MAE) seront affichés dans la console et enregistrés dans `results.txt`. Le fichier de soumission sera créé dans `submissions/submission_001_baseline.csv`.

## Méthodologie

Nous adoptons une approche itérative :
1.  **Baseline Robuste** : Mise en place d'une pipeline propre (imputation + encodage) avec un modèle rapide (HistGradientBoosting) pour établir un score de référence sans fuite de données (data leakage).
2.  **Analyses & Améliorations** : Utilisation de notebooks pour explorer les données, visualiser les résultats et tester de nouvelles hypothèses.
3.  **Suivi** : Chaque expérience est journalisée dans `experiments/experiments.md`.

## Résultats

Voir `reports/report.md` pour le détail des performances et des expériences.
