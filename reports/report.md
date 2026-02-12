# Project Report: House Prices - Advanced Regression Techniques

## Objectif
Prédire le prix de vente des maisons (SalePrice) en minimisant la Mean Absolute Error (MAE).

## Description des Données
- **Train**: `Data/train.csv` (1460 lignes, 81 colonnes)
- **Test**: `Data/test.csv` (1459 lignes, 80 colonnes - sans SalePrice)
- **Cible**: `SalePrice`
- **Features**: Mélange de variables numériques et catégorielles.

## Stratégie de Validation
- **Métrique**: MAE (Mean Absolute Error).
- **Validation**: K-Fold Cross-Validation (5 folds, shuffle=True, random_state=42).

## Historique des Runs

### RUN 001: Baseline
- **Date**: 2026-02-12
- **Description**: Modèle baseline robuste sans feature engineering complexe.
- **Pipeline**:
    - Imputation: Median (num), Most Frequent (cat).
    - Encodage: OneHotEncoder (cat).
    - Modèle: HistGradientBoostingRegressor (random_state=42).
- **Résultats CV**: 
    - MAE Moyenne = 16918.83 (+/- 1725.70)
    - MAPE Moyenne = 9.74% (+/- 1.05%)
- **Observations**: 
    - Premier jet pour établir une baseline.
    - Utilisation de `HistGradientBoostingRegressor` qui gère nativement les NaN (bien que l'imputer soit présent par sécurité et uniformité).
    - `OneHotEncoder` avec `handle_unknown='ignore'` pour éviter les erreurs sur le test set.

### RUN 002: Log Target Boosting + XGBoost
- **Date**: 2026-02-12
- **Motivation**: La distribution des prix est asymétrique (skewed). Une transformation `log1p` permet de normaliser la cible et de réduire l'impact des outliers.
- **Méthode**:
    - Transformation cible: `np.log1p(SalePrice)` entraînement -> `np.expm1(pred)` prédiction.
    - Modèle: `XGBRegressor` (n_estimators=5000, lr=0.03, max_depth=4).
- **Résultats CV**:
    - MAE Moyenne = **15093.34** (+/- 1349.12)
    - **Amélioration**: -1825$ (~10.8%) par rapport à la baseline.
- **Observations**:
    - L'approche log-target est très efficace.
    - XGBoost semble bien performer avec ces hyperparamètres par défaut.

## Prochaines Étapes
- [ ] Analyser les features importance.
- [ ] Tester d'autres algorithmes (XGBoost, LightGBM).
- [ ] Feature Engineering (création de variables, gestion des outliers).
- [ ] Tuning des hyperparamètres.
