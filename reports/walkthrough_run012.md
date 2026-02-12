# RUN 012 — Stacking Optimisé : Raisonnement de A à Z

> **Score Kaggle final : 12 882**  
> Progression totale depuis la baseline : **16 918 → 12 882 (−23.9%)**

---

## Table des matières
1. [Vue d'ensemble du pipeline](#1-vue-densemble-du-pipeline)
2. [Chargement des données](#2-chargement-des-données)
3. [Suppression des outliers](#3-suppression-des-outliers)
4. [Transformation de la cible](#4-transformation-de-la-cible-log1p)
5. [Nettoyage des données](#5-nettoyage-des-données-clean_data)
6. [Encodage ordinal](#6-encodage-ordinal-encode_ordinals)
7. [Feature Engineering](#7-feature-engineering-add_features)
8. [Target Encoding OOF](#8-target-encoding-oof)
9. [Suppression des features parasites](#9-suppression-des-features-parasites)
10. [Correction de skewness](#10-correction-de-skewness)
11. [Préprocessing des modèles](#11-preprocessing-des-modèles)
12. [Les 4 modèles de base](#12-les-4-modèles-de-base)
13. [Génération des prédictions OOF](#13-génération-des-prédictions-oof)
14. [Stacking — Le méta-modèle](#14-stacking--le-méta-modèle)
15. [Résultats et analyse](#15-résultats-et-analyse)

---

## 1. Vue d'ensemble du pipeline

```
Train.csv + Test.csv
        │
        ▼
┌─────────────────────┐
│ Suppression Outliers │  (Train uniquement)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    log1p(SalePrice)  │  Transformation cible
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   clean_data()       │  Imputation NaN (None/0/Mode)
│   encode_ordinals()  │  Qualité → numérique (Ex=5...Po=1)
│   add_features()     │  12+ features d'interaction
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Target Encoding OOF  │  Neighborhood → moyenne cible
│ Feature Drop         │  Utilities, Street, PoolQC...
│ Fix Skewness         │  log1p sur features asymétriques
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────────┐
│           4 Modèles de Base (OOF)           │
│  XGBoost │ CatBoost │ ExtraTrees │ LightGBM │
└─────────┬───────────────────────────────────┘
          ▼
┌─────────────────────┐
│ Méta-Modèle         │  ElasticNet (L1 + L2)
│ (Stack Layer 2)     │  → Combinaison optimale
└─────────┬───────────┘
          ▼
      Prédictions finales
```

### Pourquoi le Stacking ?
Un seul modèle capte un *type* de pattern. Le Stacking combine **les forces complémentaires** de chaque modèle :
- **XGBoost** : excellent sur les interactions complexes (level-wise)
- **CatBoost** : gère nativement les catégorielles, robuste au bruit
- **ExtraTrees** : randomness extrême → diversité, capte le non-linéaire
- **LightGBM** : rapide, leaf-wise, complémentaire de XGBoost

Le méta-modèle apprend à **pondérer** chaque modèle selon sa fiabilité.

---

## 2. Chargement des données

```python
train_df = pd.read_csv("Data/train.csv")   # 1460 maisons
test_df  = pd.read_csv("Data/test.csv")     # 1459 maisons
```

**Dataset** : Ames Housing (Dean De Cock, 2011). 79 variables décrivant les caractéristiques de maisons résidentielles à Ames, Iowa.

**Cible** : `SalePrice` — prix de vente en dollars.

---

## 3. Suppression des outliers

```python
# cleaning.py → remove_outliers()
outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
df = df.drop(outliers.index)   # 2 maisons supprimées
```

**Raisonnement** : L'auteur du dataset (Dean De Cock) recommande explicitement de supprimer ces 2 points. Ce sont des maisons avec une surface habitable > 4000 sqft vendues à un prix anormalement bas (< 300k$). Elles sont probablement des ventes non-bras-de-fer (foreclosures, ventes familiales) qui polluent le signal.

**Résultat** : 1460 → 1458 observations.

> [!IMPORTANT]
> Les outliers ne sont supprimés que du **train set**, jamais du test set. Le test set reste intact.

---

## 4. Transformation de la cible (log1p)

```python
y_log = np.log1p(train_df["SalePrice"])
```

**Pourquoi ?** La distribution de `SalePrice` est **fortement asymétrique à droite** (skewed) — beaucoup de maisons entre 100-200k$, peu au-dessus de 400k$.

La transformation `log1p(x) = log(1 + x)` a deux effets :
1. **Normalise la distribution** → les modèles linéaires (Ridge, ElasticNet) performent mieux
2. **Pénalise proportionnellement** → une erreur de 10k$ sur une maison de 100k$ pèse autant qu'une erreur de 30k$ sur une maison de 300k$. C'est le comportement souhaité (MAPE-like).

Les prédictions finales sont ramenées à l'échelle originale via `np.expm1()` (inverse de log1p).

---

## 5. Nettoyage des données (`clean_data`)

Le nettoyage est **domain-driven** : on ne fait pas d'imputation aveugle, on utilise la **documentation du dataset** pour comprendre ce que chaque NaN signifie.

### 5.1 — NaN = "Pas d'équipement" → `"None"`
```python
none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", ...]
```
Si une maison n'a **pas de garage**, toutes les colonnes Garage* sont NaN. Ce n'est pas une valeur manquante, c'est une information : **"pas de garage"**. On remplace par `"None"` pour le rendre explicite.

### 5.2 — NaN = 0 (surface/compteur inexistant)
```python
zero_cols = ["GarageYrBlt", "GarageArea", "GarageCars", 
             "BsmtFinSF1", "TotalBsmtSF", "MasVnrArea", ...]
```
Même logique : pas de garage → surface garage = 0.

### 5.3 — NaN = valeur manquante réelle → Imputation par Mode
```python
mode_cols = ["MSZoning", "Electrical", "KitchenQual", 
             "Exterior1st", "Exterior2nd", "SaleType"]
```
Ces quelques valeurs manquantes sont des **erreurs de saisie**. On impute par la valeur la plus fréquente.

### 5.4 — LotFrontage : Imputation groupée
```python
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
```
Le frontage (largeur du terrain sur la rue) dépend du **quartier**. On impute par la médiane du quartier, ce qui est plus précis qu'une médiane globale.

### 5.5 — Correction de type
```python
df["MSSubClass"] = df["MSSubClass"].astype(str)
```
`MSSubClass` (ex: 20, 60, 120) est un **code**, pas un nombre. "60" n'est pas 3× "20". On le convertit en string pour qu'il soit traité comme une catégorie.

---

## 6. Encodage ordinal (`encode_ordinals`)

### Le problème du One-Hot Encoding
Par défaut, les variables catégorielles sont encodées en **OneHot** : `ExterQual` crée 5 colonnes (Ex, Gd, TA, Fa, Po) avec des 0/1.

Mais `ExterQual` est **ordinale** : Ex > Gd > TA > Fa > Po. Le OneHot **perd cette information d'ordre**. Un arbre de décision ne sait pas que "Ex" est meilleur que "Gd".

### La solution : Encodage numérique ordonné
```python
qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
```

On convertit **15+ variables** en numériques ordonnés :

| Variable | Mapping | Signification |
|---|---|---|
| `ExterQual/Cond` | Po→1, Fa→2, TA→3, Gd→4, Ex→5 | Qualité/état extérieur |
| `BsmtQual/Cond` | idem + None→0 | Qualité sous-sol |
| `KitchenQual` | idem | Qualité cuisine |
| `HeatingQC` | idem | Qualité chauffage |
| `FireplaceQu` | idem | Qualité cheminée |
| `GarageQual/Cond` | idem | Qualité garage |
| `BsmtExposure` | No→1, Mn→2, Av→3, Gd→4 | Exposition sous-sol |
| `BsmtFinType1/2` | Unf→1...GLQ→6 | Type de finition sous-sol |
| `GarageFinish` | Unf→1, RFn→2, Fin→3 | Finition garage |
| `Functional` | Sal→1...Typ→8 | Fonctionnalité maison |
| `PavedDrive` | N→0, P→1, Y→2 | Allée pavée |
| `CentralAir` | N→0, Y→1 | Climatisation centrale |
| `LotShape` | IR3→0...Reg→3 | Forme du terrain |
| `LandSlope` | Sev→0, Mod→1, Gtl→2 | Pente du terrain |

**Impact** : Les modèles à base d'arbres peuvent maintenant faire des splits sur `ExterQual > 3` (= meilleur que moyen), ce qui était impossible avec le OneHot.

---

## 7. Feature Engineering (`add_features`)

On crée **12+ nouvelles features** qui capturent des relations que les modèles ne peuvent pas découvrir seuls (ou difficilement).

### 7.1 — Surfaces agrégées
```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
TotalOutdoorSF = WoodDeckSF + TotalPorchSF
```
**Pourquoi ?** Un acheteur regarde la **surface totale**, pas chaque étage séparément. La somme est un signal plus direct.

### 7.2 — Compteurs agrégés
```python
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
```
**Pourquoi ?** Un half-bath (lavabo + WC) vaut ~50% d'un full bath en terme de valeur ajoutée.

### 7.3 — Âge et rénovation
```python
Age = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
```
**Pourquoi ?** L'année de construction brute est moins informative que l'**âge au moment de la vente**. Une maison rénovée récemment (RemodAge ≈ 0) vaut plus.

### 7.4 — Indicateurs binaires
```python
HasPool, HasGarage, HasBasement, HasFireplace
```
**Pourquoi ?** La **présence** d'un équipement est un signal fort, indépendamment de sa taille. `HasPool = 1` dit "cette maison se distingue".

### 7.5 — Interactions Qualité × Surface (les top features)
```python
QualSF = OverallQual * TotalSF
QualFinishSF = OverallQual * GrLivArea
OverallScore = OverallQual * OverallCond
```
**Pourquoi ?** C'est **LA** technique la plus puissante sur ce dataset. Une grande maison (TotalSF = 3000) de qualité moyenne (Qual = 5) vaut moins qu'une maison de taille moyenne (TotalSF = 1500) de qualité excellente (Qual = 10). Le produit `QualSF = 5*3000 = 15000` vs `10*1500 = 15000` montre qu'elles se valent — c'est exactement la réalité du marché.

### 7.6 — Scores composites (utilisent l'encodage ordinal)
```python
BsmtScore = BsmtQual(encoded) * TotalBsmtSF
GarageScore = GarageQual(encoded) * GarageArea
ExterScore = ExterQual(encoded) * TotalSF
```
**Pourquoi ?** Un sous-sol de qualité "Excellent" (5) × 1000 sqft vaut bien plus qu'un sous-sol "Fair" (2) × 1000 sqft. Ces scores capturent la **valeur ajoutée combinée** qualité + taille.

### 7.7 — Ratios
```python
BsmtFinRatio = BsmtFinSF1 / TotalBsmtSF     # % du sous-sol fini
LivAreaRatio = GrLivArea / LotArea           # densité d'habitation
```
**Pourquoi ?** Le ratio de sous-sol fini indique le niveau de "finition" de la maison. La densité d'habitation capte si c'est un petit terrain bien exploité ou un grand terrain sous-utilisé.

---

## 8. Target Encoding OOF

### Le concept
Pour les variables catégorielles à **haute cardinalité** (Neighborhood = 25 catégories, Exterior1st = 15 catégories), le OneHot crée beaucoup de colonnes sparse. Le **Target Encoding** remplace chaque catégorie par la **moyenne du prix** (log) pour cette catégorie.

```
Neighborhood "StoneBr"  →  12.45  (= moyenne log(SalePrice) pour StoneBr)
Neighborhood "MeadowV"  →  11.55  (= quartier moins cher)
```

### Le problème du data leakage
Si on calcule la moyenne sur tout le train set, on **fuite** de l'information : le modèle voit indirectement la cible. C'est du data leakage.

### La solution : OOF (Out-Of-Fold) Target Encoding
```python
kf = KFold(n_splits=5)

for train_idx, val_idx in kf.split(train_df, target):
    # Calculer la moyenne UNIQUEMENT sur le train fold
    means = target[train_idx].groupby(train_df[col][train_idx]).mean()
    # Appliquer cette moyenne au val fold
    train_df.loc[val_idx, new_col] = train_df[col][val_idx].map(means)
```

**Principe** : Pour chaque observation, la moyenne est calculée **sans utiliser cette observation**. C'est le même principe que la validation croisée. Pas de leakage.

**Pour le test set** : On utilise la moyenne globale (tout le train) car on ne connaît pas la cible du test.

### Variables target-encodées
| Variable | Cardinalité | Justification |
|---|---|---|
| `Neighborhood` | 25 | Signal le plus fort du dataset (localisation !) |
| `Condition1` | 9 | Proximité routes/voies ferrées → impact prix |
| `Exterior1st` | 15 | Type de revêtement extérieur |
| `Exterior2nd` | 16 | Revêtement secondaire |

---

## 9. Suppression des features parasites

```python
DROP_FEATURES = ["Utilities", "Street", "PoolQC", "PoolArea", 
                 "MiscFeature", "MiscVal", "Id"]
```

| Feature | Raison de suppression |
|---|---|
| `Utilities` | **Quasi-constant** : 99.9% des maisons ont "AllPub" |
| `Street` | **Quasi-constant** : 99.6% ont "Pave" |
| `PoolQC` | **99% NaN** : trop peu de maisons avec piscine |
| `PoolArea` | Quasi nul pour 99% des observations |
| `MiscFeature` | Très rare, bruit pur |
| `MiscVal` | Valeur de MiscFeature, même problème |
| `Id` | Identifiant, aucun signal prédictif |

**Raisonnement** : Une feature quasi-constante n'apporte aucune information discriminante. Pire, elle peut **ajouter du bruit** et dégrader la généralisation (surtout avec les arbres profonds qui pourraient overfitter sur ces rares cas).

---

## 10. Correction de skewness

```python
from scipy.stats import skew

skewed_feats = df[numeric_cols].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats.abs() > 0.75]

for col in skewed_feats.index:
    if (df[col].dropna() >= 0).all():
        df[col] = np.log1p(df[col])
```

**Pourquoi ?** Beaucoup de features de surface (`LotArea`, `GrLivArea`, `TotalBsmtSF`) ont une distribution très **asymétrique** (skewness > 0.75). Quelques maisons ont des valeurs extrêmes qui "écrasent" la distribution.

La transformation `log1p` **compresse** les grandes valeurs et **étale** les petites. Cela aide :
- Le **Ridge/ElasticNet** (méta-modèle linéaire) : les données log-normales sont plus "linéaires"
- Les **arbres** : les splits deviennent plus équilibrés

> [!NOTE]
> On ne transforme que les features avec des valeurs ≥ 0 (pour éviter les erreurs de log sur les négatifs).

---

## 11. Preprocessing des modèles

```python
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])
```

Ce preprocessor est utilisé par **XGBoost, ExtraTrees et LightGBM** (pas CatBoost, qui gère les catégorielles nativement).

| Étape | Cible | Action |
|---|---|---|
| `SimpleImputer(median)` | Numériques | Remplit les derniers NaN par la médiane |
| `SimpleImputer(most_frequent)` | Catégorielles | Remplit par le mode |
| `OneHotEncoder(handle_unknown="ignore")` | Catégorielles | Crée des colonnes 0/1 |

`handle_unknown="ignore"` : Si le test set contient une catégorie jamais vue en train, elle est ignorée (pas d'erreur).

---

## 12. Les 4 modèles de base

### A. XGBoost — Le champion du Gradient Boosting
```python
XGBRegressor(
    n_estimators=8000,       # Beaucoup d'arbres (early stopping coupe)
    learning_rate=0.008,     # Pas de convergence très petits
    max_depth=4,             # Arbres peu profonds = moins d'overfitting
    subsample=0.7,           # 70% des données par arbre (bagging)
    colsample_bytree=0.6,    # 60% des features par arbre
    reg_alpha=0.1,           # Régularisation L1 (lasso)
    reg_lambda=2.0,          # Régularisation L2 (ridge)
    min_child_weight=3,      # Min 3 observations par feuille
    gamma=0.01,              # Gain minimum pour split
    early_stopping_rounds=150
)
```

**Philosophie du tuning** :
- **learning_rate bas** (0.008) + **n_estimators élevé** (8000) : convergence fine. L'early stopping arrête avant l'overfitting.
- **max_depth=4** : arbres peu profonds = chaque arbre est un "weak learner" → meilleure généralisation.
- **subsample + colsample** < 1 : randomisation qui réduit la corrélation entre arbres.
- **reg_alpha + reg_lambda** : double régularisation pour shrink les poids.

### B. CatBoost — Le spécialiste des catégorielles
```python
CatBoostRegressor(
    loss_function="MAE",     # Optimise directement le MAE
    iterations=6000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=5,           # Régularisation forte
    bagging_temperature=0.5, # Contrôle la randomisation
    random_strength=0.5      # Réduit le bruit dans les splits
)
```

**Avantages uniques de CatBoost** :
- **Ordered Target Statistics** : target encoding interne sans leakage (utilise l'ordre des observations)
- **Symmetric trees** : chaque split est le même à chaque profondeur → plus rapide et plus régulier
- **loss_function="MAE"** : optimise directement la métrique Kaggle

### C. ExtraTrees — La diversité par le hasard
```python
ExtraTreesRegressor(
    n_estimators=800,
    max_depth=20,            # Plus profond (pas de boosting)
    min_samples_split=3,
    min_samples_leaf=2
)
```

**Différence avec Random Forest** : les ExtraTrees choisissent les **seuils de split aléatoirement** au lieu de chercher le meilleur seuil. Cela crée une **diversité extrême** qui complète les modèles de boosting.

**Rôle dans le stack** : apporter un signal "orthogonal" aux boosters (XGB, Cat, LGB).

### D. LightGBM — Le complémentaire rapide
```python
LGBMRegressor(
    n_estimators=8000,
    learning_rate=0.008,
    max_depth=4,
    num_leaves=20,           # Moins de feuilles = moins d'overfitting
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=0.2,           # L1 plus fort
    reg_lambda=2.0,
    min_child_samples=10     # Min 10 obs par feuille
)
```

**Différence avec XGBoost** : LightGBM construit les arbres **leaf-wise** (feuille par feuille, la plus informative d'abord) au lieu de **level-wise** (niveau par niveau). Cela capte des patterns différents.

> [!NOTE]
> Au final, l'ElasticNet a mis le coefficient de LGB à **0.00**, signifiant que dans ce contexte spécifique, le signal de LGB était déjà capté par XGB+CatBoost+ET. C'est le L1 (Lasso) qui a fait cette sélection automatique.

---

## 13. Génération des prédictions OOF

### Le processus Out-Of-Fold (OOF)

```
                           K-Fold (K=5)
    ┌────────────────────────────────────────────┐
    │  Fold 1   Train: [2,3,4,5]  Val: [1]      │  → pred_1
    │  Fold 2   Train: [1,3,4,5]  Val: [2]      │  → pred_2
    │  Fold 3   Train: [1,2,4,5]  Val: [3]      │  → pred_3
    │  Fold 4   Train: [1,2,3,5]  Val: [4]      │  → pred_4
    │  Fold 5   Train: [1,2,3,4]  Val: [5]      │  → pred_5
    └────────────────────────────────────────────┘
    
    OOF_train = [pred_1, pred_2, pred_3, pred_4, pred_5]  (1458 prédictions)
    OOF_test  = moyenne(test_pred_fold1, ..., test_pred_fold5)
```

**Pourquoi OOF ?**
1. Chaque prediction du train est faite par un modèle qui **n'a jamais vu cette observation** → pas d'overfitting
2. On obtient des prédictions "honnêtes" pour **TOUT** le train set
3. Ces prédictions deviennent les **features du méta-modèle**

**Pour le test set** : Chaque fold produit une prédiction du test. On prend la **moyenne des 5** pour réduire la variance.

### Early Stopping (XGBoost et LightGBM)
```python
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # Monitore la perf sur val
    verbose=False
)
# early_stopping_rounds=150 → Si pas d'amélioration pendant 150 rounds, STOP
```

Au lieu de fixer un nombre d'arbres arbitraire, on laisse le modèle s'entraîner jusqu'à ce qu'il **arrête de progresser** sur la validation. Typiquement, XGB utilise ~1000-2000 des 8000 arbres possibles.

---

## 14. Stacking — Le méta-modèle

### L'architecture
```
                    Couche 1 (4 modèles de base)
                    ┌──────────────────────────────────┐
                    │  OOF_XGB  OOF_Cat  OOF_ET  OOF_LGB │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    Couche 2 (méta-modèle)
                    ┌──────────────────────────────────┐
                    │         ElasticNet                │
                    │   0.58*XGB + 0.34*Cat + 0.10*ET  │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                           Prédiction finale
```

### Pourquoi ElasticNet plutôt que Ridge ?

| Meta-Model | MAE CV | Technique |
|---|---|---|
| Ridge (L2 seul) | 13 796 | Garde tous les modèles, shrink les poids |
| **ElasticNet (L1+L2)** | **13 768** | Peut **éliminer** un modèle (coef → 0) |

L'ElasticNet combine la régularisation **L1 (Lasso)** et **L2 (Ridge)** :
- **L1** : force certains coefficients à zéro → **sélection de modèles**
- **L2** : shrink les coefficients → **stabilité**

```python
ElasticNet(alpha=0.001, l1_ratio=0.5)
# alpha = force globale de régularisation
# l1_ratio = 0.5 → 50% L1 + 50% L2
```

### Les coefficients appris

| Modèle | Coefficient | Interprétation |
|---|---|---|
| **XGBoost** | **0.5765** | Modèle dominant, le plus fiable |
| **CatBoost** | **0.3424** | Complémentaire, capte ce que XGB manque |
| **ExtraTrees** | **0.1020** | Quelques % de diversité utile |
| **LightGBM** | **0.0000** | Éliminé (redondant avec XGB) |

**Analyse** : Le L1 a déterminé que LightGBM n'apportait **aucune information supplémentaire** par rapport à XGB + Cat + ET. Les deux boosters (XGB et LGB) étant construits sur des arbres GBDT, leurs erreurs sont fortement corrélées. Le méta-modèle a automatiquement éliminé le plus faible.

---

## 15. Résultats et analyse

### Métriques finales

| Métrique | Valeur |
|---|---|
| **MAE CV (OOF)** | 13 768 |
| **MAPE CV** | 7.88% |
| **Score Kaggle** | **12 882** |

### Progression à travers les runs

| Run | Méthode | Kaggle | Δ |
|---|---|---|---|
| 001 | Baseline Ridge | 16 918 | — |
| 003 | XGB + FE + Tuning | 14 125 | −2 793 |
| 008 | XGB + Cleaning | 13 692 | −3 226 |
| 009 | Stacking 3 modèles | 13 375 | −3 543 |
| 010 | + Ordinals + LGB | 13 075 | −3 843 |
| 011 | + Target Encoding | 12 970 | −3 948 |
| **012** | **+ Tuning + ElasticNet** | **12 882** | **−4 036** |

### Ce qui a le plus contribué au score final

```
                  Impact estimé par étape
  ┌────────────────────────────────────────────┐
  │  Feature Engineering (TotalSF, QualSF)     │ ████████████  ~30%
  │  Stacking (multi-model)                    │ ██████████    ~25%
  │  Target Encoding (Neighborhood)            │ ████████      ~15%
  │  Ordinal Encoding                          │ ██████        ~10%
  │  Outlier Removal                           │ █████         ~8%
  │  Hyperparameter Tuning                     │ ████          ~7%
  │  Skewness + Feature Drop                   │ ███           ~5%
  └────────────────────────────────────────────┘
```

### Leçons retenues

1. **Feature Engineering > Tuning** : Créer `QualSF = OverallQual × TotalSF` a plus d'impact que des heures de tuning.
2. **L'encodage ordinal est crucial** : Les variables qualité contiennent un **ordre** que le OneHot détruit.
3. **Le Target Encoding OOF est puissant** : Transformer Neighborhood en "prix moyen du quartier" donne un signal direct que 25 colonnes OneHot ne peuvent pas offrir.
4. **L'ElasticNet sélectionne** : Le L1 élimine automatiquement les modèles redondants.
5. **Le CV ne prédit pas toujours le Kaggle** : RUN 010 avait un meilleur CV que RUN 011, mais un Kaggle pire. La généralisation ≠ le fit sur le train.
6. **Le nettoyage domain-driven** est essentiel : Comprendre que NaN dans `BsmtQual` signifie "pas de sous-sol" (et pas une donnée manquante) change tout.
