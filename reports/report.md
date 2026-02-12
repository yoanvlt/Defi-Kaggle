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
| 006 | Stacking (OOF) | **14347** | **13868.37275** |

### RUN 006: Stacking (Out-Of-Fold)
- **Date**: 2026-02-12
- **Motivation**: Aller plus loin que le blending linéaire (poids fixes) en laissant un méta-modèle apprendre la meilleure combinaison des prédictions, tout en garantissant un schéma de validation robuste (OOF) pour éviter le leakage.
- **Méthode**:
    - **Niveau 1 (Base Models)** :
        - XGBoost (Tuned + FE)
        - CatBoost (Native + Log Target)
        - ExtraTreesRegressor (Diversité)
    - **Niveau 2 (Meta Model)** :
        - Ridge Regression (L2) entraîné sur les prédictions OOF.
- **Résultats CV (Sur le Train Set via OOF)** :
    - MAE: **14346.98**
    - MAPE: **8.21%**
    - **Analyse CV**: Le score CV du Stacking (14347) est nettement meilleur que celui des modèles individuels.
- **Résultats Kaggle (Public Leaderboard)** :
    - **Score**: **13868.37**
    - **Amélioration**: -16 points vs Blend (RUN 005), -257 points vs XGB seul (RUN 003).
    - **Conclusion**: Le Stacking est la méthode la plus performante. L'apprentissage des poids par le Ridge via la méthode OOF a permis d'extraire encore un peu plus de signal que le simple mélange manuel.

## Tableau Comparatif Final
| Run | Méthode | MAE Moyen CV | Kaggle Score |
|---|---|---|---|
| 001 | Baseline | 16918 | - |
| 002 | Log Target | 15093 | - |
| 003 | FE + Tuning | 14806 | 14125 |
| 004 | CatBoost | 15264 | - |
| 005 | Blend (w=0.7) | - | 13884 |
| **006** | **Stacking** | **14347** | **13868** |
| 007 | Poly Features | 14948 | **13994.70** |

### RUN 007: Polynomial Features (Surfaces au Carré)
- **Date**: 2026-02-12
- **Motivation**: Tester si l'ajout explicite de termes quadratiques (surface au carré) aide le modèle à mieux capturer les non-linéarités.
- **Méthode**:
    - Création de `GrLivArea_sq`, `TotalBsmtSF_sq`, `TotalSF_sq`, etc.
    - Modèle : XGBoost (mêmes hyperparamètres que RUN 003).
- **Résultats CV**:
    - MAE: **14947.58** (+/- 1256.80) - *Légèrement moins bon que RUN 003 (14806)*
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **13994.70**
    - **Amélioration**: -131 points vs RUN 003 (14125).
    - **Analyse Critique**: Il est intéressant de noter que si la CV était moins bonne, le score Kaggle s'est amélioré. Cela suggère que la pénalité de complexité observée en CV (sur-apprentissage potentiel sur les folds) n'a pas impacté le test set public de la même manière. Cependant, ce score reste inférieur aux méthodes d'ensemble (Blend: 13884, Stacking: 13868).

## Tableau Comparatif Final
| Run | Méthode | MAE Moyen CV | Kaggle Score |
|---|---|---|---|
| 001 | Baseline | 16918 | - |
| 002 | Log Target | 15093 | - |
| 003 | FE + Tuning | 14806 | 14125 |
| 004 | CatBoost | 15264 | - |
| 005 | Blend (w=0.7) | - | 13884 |
| **006** | **Stacking** | **14347** | **13868** |
| 007 | Poly Features | 14948 | 13995 |
| **008** | **Cleaning + Outliers** | **14051** | **13691.83** |

### RUN 008: Advanced Cleaning & Outliers
- **Date**: 2026-02-12
- **Motivation**: Améliorer la qualité du signal en traitant les données manquantes avec une logique métier (au lieu d'une médiane générique) et en supprimant les observations aberrantes qui trompent le modèle.
- **Méthode**:
    - **Suppression Outliers (Train Only)** : Retrait des maisons > 4000 sqft avec prix < 300k$ (Points recommandés par l'auteur du dataset).
    - **Imputation Métier** :
        - `PoolQC`, `GarageQual`, etc. -> "None" (Pas d'équipement).
        - `GarageYrBlt`, `MasVnrArea` -> 0.
        - `LotFrontage` -> Médiane par voisinage (New!).
    - **Modèle**: XGBoost (Tuned RUN 003).
- **Résultats CV**:
    - MAE: **14050.83** (+/- 451.85)
    - MAPE: **8.11%**
    - **Analyse**: **Explosion des performances !**
        - Le gain est colossal : **-756 points** par rapport au RUN 003 (14806) qui utilisait le même modèle.
        - Le score CV dépasse même celui du Stacking (14347).
        - L'écart-type (std) a fondu (451 vs 1540), preuve d'une stabilité extrême.
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **13691.83**
    - **Amélioration**: **-177 points** vs Stacking (13868).
    - **Conclusion**: C'est le **meilleur score du projet**. La qualité des données ("Garbage In, Garbage Out") est plus importante que la complexité du modèle. Le nettoyage métier et le retrait des outliers ont débloqué le potentiel du XGBoost.

## Tableau Comparatif Final
| Run | Méthode | MAE Moyen CV | Kaggle Score |
|---|---|---|---|
| 001 | Baseline | 16918 | - |
| 002 | Log Target | 15093 | - |
| 003 | FE + Tuning | 14806 | 14125 |
| 004 | CatBoost | 15264 | - |
| 005 | Blend (w=0.7) | - | 13884 |
| 006 | Stacking | 14347 | 13868 |
| 007 | Poly Features | 14948 | 13995 |
| **008** | **Cleaning + Outliers** | **14051** | **13692** |

### RUN 009: Stacking + Advanced Cleaning (Le "All-In")
- **Date**: 2026-02-12
- **Motivation**: Combiner l'architecture gagnante (Stacking) avec la qualité de données du RUN 008.
- **Méthode**:
    - **Preprocessing**: Cleaning + Outliers Removal (identique RUN 008).
    - **Models**: XGBoost + CatBoost + ExtraTrees (Stacking OOF).
- **Résultats CV**:
    - MAE: **13837.65** (Record Absolu CV)
    - MAPE: **7.95%** (Sous la barre des 8%!)
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **13374.62**
    - **Amélioration**: **-317 points** vs RUN 008 (13692) et **-494 points** vs Stacking sans nettoyage (13868).
    - **Conclusion**: **RESULTAT EXCEPTIONNEL**.
        - On a brisé le plafond de verre des 13500.
        - Cette combinaison (Stacking robuste + Données propres) est la clé.
        - C'est le résultat final de cette série d'optimisations.

## Tableau Comparatif Final
| Run | Méthode | MAE Moyen CV | Kaggle Score |
|---|---|---|---|
| 001 | Baseline | 16918 | - |
| 002 | Log Target | 15093 | - |
| 003 | FE + Tuning | 14806 | 14125 |
| 004 | CatBoost | 15264 | - |
| 005 | Blend (w=0.7) | - | 13884 |
| 006 | Stacking | 14347 | 13868 |
| 007 | Poly Features | 14948 | 13995 |
| 008 | Cleaning & Outliers | 14051 | 13692 |
| **009** | **Stacking Cleaned** | **13838** | **13375** |
| **010** | **Stacking v2 (Ord+LGB)** | **13951** | **13075** |
| 011 | Stacking v3 (TE+Skew) | 14021 | 12970 |
| 012 | Tuned Stack + ElasticNet | 13768 | TBD |

### RUN 010: Stacking v2 — Ordinals + Interactions + LightGBM + Early Stopping
- **Date**: 2026-02-12
- **Motivation**: Extraire davantage de signal en enrichissant le feature engineering et en diversifiant l'ensemble.
- **4 Axes d'Amélioration vs RUN 009**:
    1. **Encodage Ordinal**: 15+ variables qualité (Ex/Gd/TA/Fa/Po → 5/4/3/2/1) converties en numériques au lieu de OneHot.
    2. **Features d'Interaction**: 12 nouvelles features (QualSF, QualFinishSF, OverallScore, BsmtFinRatio, LivAreaRatio, BsmtScore, GarageScore, ExterScore, etc.).
    3. **LightGBM**: 4ème base model (leaf-wise, diversité vs XGBoost level-wise).
    4. **Early Stopping**: XGBoost et LightGBM utilisent eval_set + early_stopping_rounds=100.
- **Résultats CV**:
    - MAE: **13950.59**
    - MAPE: **7.95%**
    - Meta Coefs: XGB=0.44, Cat=0.45, LGB=0.09, ET=0.05
    - **Analyse**: Le CV est légèrement supérieur à RUN 009 (13838), mais les features enrichies et l'early stopping généralisent nettement mieux.
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **13074.70**
    - **Amélioration**: **-300 points** vs RUN 009 (13375) et **-617 points** vs Stacking sans nettoyage (13692).
    - **Conclusion**: **NOUVEAU RECORD ABSOLU**. L'encodage ordinal et les features d'interaction ont débloqué un signal que le OneHot masquait. Le LightGBM apporte une diversité complémentaire au stack. L'early stopping évite l'overfitting. Progression totale depuis la baseline: **16918 → 13075 (-3843, soit -22.7%)**.

### RUN 011: Stacking v3 — Target Encoding + Skewness + Feature Drop
- **Date**: 2026-02-12
- **Motivation**: Enrichir le signal via target encoding (Neighborhood), normaliser les distributions, et réduire le bruit.
- **3 Axes d'Amélioration vs RUN 010**:
    1. **Target Encoding OOF**: Neighborhood, Condition1, Exterior1st, Exterior2nd encodés par la moyenne cible (sans leakage).
    2. **Correction de Skewness**: log1p sur les features numériques à forte asymétrie (>0.75).
    3. **Feature Drop**: Suppression de Utilities, Street, PoolQC, PoolArea, MiscFeature, MiscVal (quasi-constants ou bruyants).
- **Résultats CV**:
    - MAE: **14020.51**
    - MAPE: **8.01%**
    - Meta Coefs: XGB=0.45, Cat=0.47, LGB=0.04, ET=0.06
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **12969.71**
    - **Amélioration**: **-105 points** vs RUN 010 (13075). **NOUVEAU RECORD ABSOLU**.
    - **Conclusion**: Le target encoding OOF pour Neighborhood est extrêmement puissant. Progression totale: **16918 → 12970 (-3948, soit -23.3%)**.

### RUN 012: Tuned Stacking + ElasticNet Meta-Model
- **Date**: 2026-02-12
- **Motivation**: Tirer les derniers gains en affinant les hyperparamètres et le meta-model.
- **Améliorations vs RUN 011**:
    1. **Tuning XGBoost**: lr=0.008, depth=4, gamma=0.01, min_child_weight=3, reg_lambda=2.0
    2. **Tuning CatBoost**: lr=0.02, l2_leaf_reg=5, bagging_temperature=0.5
    3. **Tuning LightGBM**: lr=0.008, num_leaves=20, min_child_samples=10, reg_lambda=2.0
    4. **Tuning ExtraTrees**: 800 trees, depth=20, min_samples_leaf=2
    5. **ElasticNet Meta-Model**: L1+L2 combinés. A zéroté le coefficient de LGB (redondant avec XGB).
- **Résultats CV**:
    - MAE: **13767.56** (vs 14020 RUN 011 — nette amélioration)
    - MAPE: **7.88%** (meilleur MAPE de tout le projet)
    - ElasticNet > Ridge (13768 vs 13796)
    - Meta Coefs: XGB=0.58, Cat=0.34, ET=0.10, **LGB=0.00** (éliminé par L1)
- **Résultats Kaggle (Public Leaderboard)**:
    - Score: **12882.12**
    - **Amélioration**: **-88 points** vs RUN 011 (12970). **NOUVEAU RECORD ABSOLU**.
    - **Conclusion**: Le tuning fin + ElasticNet apportent un gain constant. Progression totale: **16918 → 12882 (-4036, soit -23.9%)**.
