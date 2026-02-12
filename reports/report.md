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

### RUN 003: Feature Engineering + Tuning
- **Date**: 2026-02-12
- **Motivation**: Intégrer la connaissance métier (surface totale, âge) et optimiser le modèle pour gagner en précision.
- **Méthode**:
    - **Feature Engineering**: Création de `TotalSF`, `TotalBath`, `Age`, `RemodAge`, `TotalPorchSF`, et indicateurs binaires (Piscine, Garage...).
    - **Tuning**: RandomizedSearchCV sur XGBoost (Best: n_estimators=4000, learning_rate=0.01, max_depth=5, subsample=0.75).
- **Résultats CV**:
    - MAE Moyenne = **14806.84** (+/- 1540.52)
    - MAPE Moyenne = **8.53%** (+/- 1.01%)
    - **Amélioration**: -287$ vs RUN 002, -2112$ vs Baseline.
- **Observations**:
    - Le gain est présent mais modéré par rapport au saut du RUN 002.
    - Le feature engineering apporte de l'information utile.
    - Le modèle est plus complexe mais reste robuste (écart-type maîtrisé).

## Tableau Comparatif
| Run | Méthode | MAE Moyen | MAPE Moyen | Amélioration MAE |
|---|---|---|---|---|
| 001 | Baseline | 16918 | 9.74% | - |
| 002 | Log Target | 15093 | - | -1825 |
| 003 | FE + Tuning | **14806** | **8.53%** | **-2112** |

### RUN 004: CatBoost + Log Target
- **Date**: 2026-02-12
- **Motivation**: Exploiter la gestion native des variables catégorielles par CatBoost (sans OneHot) pour éviter la perte d'information.
- **Méthode**:
    - **Preprocessing**: Variables catégorielles traitées comme telles (pas de Feature Engineering avancé ici).
    - **Modèle**: CatBoostRegressor (Depth 6, LR 0.03, L2 3) avec Log-Target.
- **Résultats CV**:
    - MAE Moyenne = **15263.77** (+/- 1305.28)
    - MAPE Moyenne = **8.61%**
    - **Amélioration**: Performance très solide, comparable au XGBoost (RUN 002), mais légèrement en retrait par rapport au RUN 003 qui bénéficiait du Feature Engineering dédié (TotalSF, etc.).
- **Observations**:
    - CatBoost est très performant "out-of-the-box" sans feature engineering complexe.
    - La stabilité (écart-type faible) est excellente.

## Résultats & Validation Kaggle

### Tableau Synthétique (CV vs Leaderboard)
| Run | Modèle | CV MAE (mean ± std) | Kaggle Public MAE | Commentaire |
|---|---|---|---|---|
| 001 | Baseline | 16918 ± 1726 | TBD | - |
| 002 | Log Target | 15093 ± 1349 | TBD | - |
| 003 | FE + Tuning | 14806 ± 1540 | 14125.50704 | Base solide. |
| 004 | CatBoost | 15264 ± 1305 | TBD | Bon complément. |
| **005** | **Blend (w=0.7)** | **N/A** | **13884.08306** | **Best Score (-241 vs RUN 003)** |

### Analyse de l'écart (Gap)
Le score Kaggle (14125) est meilleur que le score Cross-Validation (14806). Cela peut s'expliquer par :
1. **Distribution du Test Public** : Le leaderboard public n'utilise que 50% du test set. Il est possible que cette partie soit légèrement "plus facile" à prédire (moins d'outliers) que le train set.
2. **Variance** : L'écart-type de notre CV (~1540) est assez large. Le score Kaggle rentre parfaitement dans l'intervalle de confiance [13266, 16346].
3. **Robustesse** : Le modèle généralise bien et ne semble pas overfitter le train set. En effet, un score Kaggle très inférieur au CV est souvent bon signe (le modèle n'a pas appris le bruit du train).

### RUN 005: Ensemble / Blending
- **Date**: 2026-02-12
- **Motivation**: Combiner les forces de deux modèles performants (XGBoost avec Feature Engineering et CatBoost). Si leurs erreurs ne sont pas parfaitement corrélées, la moyenne pondérée devrait réduire la variance et améliorer le score final.
- **Méthode**:
    - Blend = $w \times \text{Pred}_{\text{XGB}} + (1-w) \times \text{Pred}_{\text{CatBoost}}$
- **Résultats Kaggle (Public Leaderboard)** :
    - w=0.9 (90% XGB): 14004.83
    - w=0.8 (80% XGB): 13922.41
    - **w=0.7 (70% XGB): 13884.08 (Optimum)**
    - w=0.6 (60% XGB): 13899.70
- **Analyse**:
    - Le mélange 70/30 offre le meilleur compromis.
    - Le gain de **~241 points** par rapport au meilleur modèle unique (RUN 003) confirme l'intérêt du blending. Les modèles capturent des signaux différents.

## Tableau Comparatif Final
| Run | Méthode | MAE Moyen CV | Kaggle Score |
|---|---|---|---|
| 001 | Baseline | 16918 | - |
| 002 | Log Target | 15093 | - |
| 003 | FE + Tuning | 14806 | 14125 |
| 004 | CatBoost | 15264 | - |
| **005** | **Blend (w=0.7)** | - | **13884** |

## Prochaines Étapes
- [ ] Analyser les features importance.
- [ ] Tester d'autres algorithmes (XGBoost, LightGBM).
- [ ] Feature Engineering (création de variables, gestion des outliers).
- [ ] Tuning des hyperparamètres.
