# Rapport final — Prédiction du prix de vente (House Prices)

## 1) Introduction
Dans le cadre de ce devoir, l’objectif était de construire un modèle de **régression** capable de prédire le prix de vente d’une maison (`SalePrice`) à partir d’un jeu de données tabulaire mélangeant des variables **numériques** et **catégorielles**.  
Le but n’était pas uniquement d’obtenir un bon score, mais surtout de **justifier une démarche d’amélioration progressive**, en comprenant ce qui améliore réellement les performances : preprocessing, transformations, feature engineering, ensembles (blend/stack), nettoyage, et enfin tuning.

- **Métrique principale :** MAE (Mean Absolute Error), car elle correspond directement à l’erreur moyenne en dollars (plus elle est basse, mieux c’est).
- **Métrique secondaire :** MAPE (pour interpréter l’erreur en %).

---

## 2) Données et protocole d’évaluation

### 2.1 Données
- **Train** : 1460 lignes, 81 colonnes (cible incluse)
- **Test** : 1459 lignes, 80 colonnes (cible absente)

Les colonnes contiennent :
- des **variables numériques** (surfaces, année de construction, etc.)
- des **variables catégorielles** (quartier, type de logement, qualité, etc.)
- des **valeurs manquantes**, parfois “réelles” (info absente) et parfois “structurelles” (ex : absence de garage).

### 2.2 Validation
Pour comparer les itérations de manière équitable, j’ai utilisé une validation :
- **KFold (5 folds)**, avec mélange des données (shuffle) et graine fixe (random_state=42)

Cette méthode me permet :
- de mesurer la performance moyenne
- de limiter le risque de surapprentissage sur un split unique
- d’avoir une base stable pour comparer mes runs

---

## 3) Méthodologie générale : logique d’itération
Ma logique a été de progresser en “couches”, du plus simple au plus avancé :

1) **Baseline stable** : créer un pipeline fonctionnel (imputation + encodage + modèle)  
2) **Gains structurels** : transformation de la cible (log)  
3) **Meilleur modèle tabulaire** : gradient boosting (XGB/Cat/LGB)  
4) **Feature engineering** : créer des variables plus explicatives  
5) **Ensembles** : blending puis stacking OOF pour réduire la variance  
6) **Nettoyage & outliers** : améliorer la qualité du signal d’entrée  
7) **Encodages avancés** : ordinal encoding, target encoding OOF  
8) **Tuning final + régularisation** : ajuster hyperparamètres + meta-modèle robuste  

À chaque run, je cherche à répondre à une question simple du type :  
> **“Qu’est-ce qui bloque la perf maintenant : le modèle, les features, ou les données ?”**

---

## 4) Résultats et analyse par runs

### RUN 001 — Baseline (référence)
**Idée :** obtenir un premier pipeline complet, robuste et reproductible.  
- Imputation simple (num: médiane / cat: mode)  
- OneHotEncoder  
- Modèle : HistGradientBoostingRegressor  

**Résultat :** MAE CV ≈ **16918**  
**Analyse :** baseline indispensable : elle sert de point zéro. À ce stade, je sais que le pipeline fonctionne mais que la perf est loin d’un bon score Kaggle.

---

### RUN 002 — Transformation log de la cible + XGBoost
**Idée :** le prix est très asymétrique (skew), donc je transforme la cible pour réduire l’effet des gros outliers.  
- `y = log1p(SalePrice)`  
- modèle plus puissant : XGBoost  

**Résultat :** MAE CV ≈ **15093**  
**Analyse :** c’est le plus gros “saut” simple. Conclusion : **la distribution de la cible** était un point clé.

---

### RUN 003 — Feature engineering + tuning XGB
**Idée :** injecter du signal “métier” au lieu de tout laisser au modèle.  
- variables agrégées (ex : surfaces totales)  
- premiers ajustements d’hyperparamètres (RandomizedSearchCV)

**Résultat :** MAE CV ≈ **14807**, MAPE ≈ **8.53%**  
**Analyse :** gain réel mais plus faible : on entre dans les gains “incrémentaux”.

---

### RUN 004 — Test CatBoost
**Idée :** CatBoost gère très bien les catégories, donc je teste si ça dépasse XGB.  
**Résultat :** MAE CV ≈ **15264**  
**Analyse :** dans mon contexte et avec mon pipeline, CatBoost est solide mais **pas meilleur** que XGB + FE.

---

### RUN 005 — Blending XGB + CatBoost
**Idée :** deux modèles forts font des erreurs différentes → moyenne pondérée = réduction variance.  
**Résultat Kaggle :** ≈ **13884**  
**Analyse :** gros gain : première preuve que **combiner des modèles** est plus efficace que “tuner un seul modèle” à l’infini.

---

### RUN 006 — Stacking OOF (Ridge meta-model)
**Idée :** au lieu de choisir un poids fixe, je laisse un méta-modèle apprendre les bons poids.  
- OOF predictions (anti-leakage)  
- meta : Ridge  
- base : XGB + CatBoost + ExtraTrees  

**Résultat :** MAE CV ≈ **14347** ; Kaggle ≈ **13868**  
**Analyse :** stacking > blending car il apprend la combinaison. La rigueur “OOF” évite d’optimiser sur des prédictions déjà vues.

---

### RUN 007 — Hypothèse surfaces² (non-linéarité explicite)
**Idée :** rendre certaines relations plus expressives (effet taille non linéaire).  
**Résultat :** MAE CV ≈ **14948** ; Kaggle ≈ **13995**  
**Analyse :** pas un gain net. Ça montre qu’une idée peut être “logique” mais ne pas performer. Intéressant aussi : parfois **LB et CV ne bougent pas pareil**.

---

### RUN 008 — Nettoyage avancé + outliers
**Idée :** arrêter de pousser le modèle si les données restent bruitées.  
- suppression outliers  
- meilleure gestion des NA : “None” = absence réelle (garage, cheminée…)  
- imputation plus cohérente (LotFrontage par voisinage, 0 pour surfaces absentes, etc.)

**Résultat :** MAE CV ≈ **14051** ; Kaggle ≈ **13692**  
**Analyse :** énorme point d’apprentissage : **la qualité des données donne plus de gain** que certains changements de modèle.

---

### RUN 009 — Stacking + nettoyage
**Idée :** combiner les deux meilleurs leviers : pipeline stacking + nettoyage.  
**Résultat :** MAE CV ≈ **13838** ; Kaggle ≈ **13375**  
**Analyse :** run “logique” : je consolide le meilleur setup au lieu de repartir dans toutes les directions.

---

### RUN 010 — Stacking v2 : ordinals + interactions + LightGBM + early stopping
**Idée :** augmenter le signal (ordinals), diversifier les modèles (LGB), limiter l’overfit (early stopping).  
**Résultat Kaggle :** ≈ **13075**  
**Analyse :** très bon gain Kaggle → amélioration de la **généralisation**.

---

### RUN 011 — Target Encoding OOF + skewness features + drop variables bruitées
**Idée :** améliorer le traitement des catégories “fortes” (ex : Neighborhood) sans exploser en OneHot.  
- Target encoding OOF (anti-leakage)  
- log1p sur features skewed  
- drop de features peu utiles/bruit

**Résultat Kaggle :** ≈ **12970**  
**Analyse :** le target encoding apporte beaucoup, surtout sur des catégories très structurantes.

---

### RUN 012 — Tuning final + ElasticNet meta-model
**Idée :** “gratter les derniers points” proprement : tuning base models + meta plus robuste (ElasticNet).  
- tuning XGB/Cat/LGB/ExtraTrees  
- ElasticNet meta : L1+L2 (sélection + stabilité)

**Résultat :** MAE CV ≈ **13768** ; Kaggle ≈ **12882**  
**Analyse :** l’ElasticNet stabilise le stack et réduit la redondance (ex : un modèle peut être sous-pondéré/éliminé si peu utile).

---

### RUN 013 — Linear Stack (Ridge + Lasso)
**Idée :** ajouter une **diversité architecturale** : modèles linéaires + trees ne font pas les mêmes erreurs.  
- Ajout Ridge et Lasso comme base models + pipeline avec scaler  
- Stack 6 modèles → méta ElasticNet

**Résultat :** MAE CV ≈ **13264**, MAPE ≈ **7.63%** ; Kaggle ≈ **12608** (**meilleur score**)  
**Analyse :** l’ajout de modèles linéaires apporte un signal orthogonal aux trees.

---

### RUN 016 — Feature Boost (10 nouvelles features métier)
**Idée :** plutôt que de pousser le nettoyage (risque d’overfit), injecter du **signal métier** (rénovation, saison, ratios, scores qualité…) surtout utile aux modèles linéaires.  
- 10 features : rénovation, densité bains/surface, score qualité global, saison de vente, ratio sous-sol aménagé, etc.  
- Stack identique RUN 013 (6 modèles) → méta ElasticNet

**Résultat :** MAE CV ≈ **13183**, MAPE ≈ **7.58%** ; Kaggle ≈ **12537** (**nouveau meilleur score**)  
**Analyse :** gain “propre” et généralisable ; les nouvelles features profitent surtout aux modèles linéaires (Lasso fortement pondéré).

---

## 5) Discussion : ce que j’ai compris grâce au projet

### 5.1 Ce qui a le plus compté
1) **Log-transform de la cible** : gros gain immédiat  
2) **Nettoyage / outliers / NA “intelligents”** : amélioration majeure du signal  
3) **Ensembles (blend/stack)** : réduction de variance et meilleure généralisation  
4) **Encodages avancés (TE OOF, ordinals)** : gain sur la structure des catégories  
5) **Diversité des modèles (trees + linéaires)** : gros palier sur RUN 013  
6) **Features métier** : gains finaux “sains” sur RUN 016

### 5.2 Ce qui a moins payé
- Les idées de features “polynomiales” seules (surfaces²) : plausibles, mais pas au niveau des ensembles + nettoyage.

### 5.3 Le compromis CV vs Kaggle
J’ai observé que certains runs améliorent plus Kaggle que la CV. Ça peut venir :
- du découpage KFold vs distribution du public test  
- de la variance du dataset (dataset relativement petit)  
- d’améliorations ciblées sur certains profils de maisons plus présents dans le test public

---

## 6) Conclusion
Au final, j’ai suivi une démarche progressive et cohérente : partir d’une baseline, corriger la distribution de la cible, enrichir les features, puis passer à des ensembles, et enfin travailler le nettoyage des données et les encodages avancés.  
Le meilleur score final (**12537** sur Kaggle public) est arrivé grâce à la **combinaison des meilleures décisions** : données propres + stacking OOF + encodage + régularisation, puis un vrai gain via la **diversité des modèles** (RUN 013) et enfin un dernier palier en ajoutant des **features métier** (RUN 016) qui apportent un signal plus explicite et plus généralisable.

---

## 7) Tableau récapitulatif des résultats

| Run | Idée principale | MAE CV (mean) | MAPE CV | Kaggle Public MAE | Observation |
|---:|---|---:|---:|---:|---|
| 001 | Baseline pipeline | 16918 | 9.74% | - | Référence |
| 002 | log-target + XGB | 15093 | - | - | Gros saut |
| 003 | FE + tuning XGB | 14807 | 8.53% | 14126 | Gain incrémental |
| 004 | CatBoost | 15264 | 8.61% | - | Stable mais < XGB |
| 005 | Blend XGB/Cat | - | - | 13884 | Réduction variance |
| 006 | Stacking OOF (Ridge) | 14347 | 8.21% | 13868 | Combo “apprise” |
| 007 | Surfaces² | 14948 | - | 13995 | Hypothèse non validée |
| 008 | Cleaning + outliers | 14051 | 8.11% | 13692 | Données > modèle |
| 009 | Stacking + cleaning | 13838 | 7.95% | 13375 | Record important |
| 010 | Ordinals + interactions + LGB + ES | 13951 | 7.95% | 13075 | Généralisation ↑ |
| 011 | TE OOF + skew + drop | 14021 | 8.01% | 12970 | TE très fort |
| 012 | Tuning + ElasticNet meta | 13768 | 7.88% | 12882 | Sélection ElasticNet |
| 013 | Linear Stack (Ridge+Lasso) | 13264 | 7.63% | 12608 | Meilleur final (avant RUN 016) |
| 016 | Feature Boost (10 features) | 13183 | 7.58% | **12537** | Meilleur final |