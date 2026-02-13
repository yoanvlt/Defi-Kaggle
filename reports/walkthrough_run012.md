# RUN 012 — Prédiction de Prix Immobilier : Notre Approche de A à Z

> **Contexte** : Compétition Kaggle — prédire le prix de maisons à partir de 79 caractéristiques.  
> **Notre meilleur score : 12 882 $** d'erreur moyenne (MAE).  
> **Amélioration : −23.9%** par rapport à notre premier modèle naïf.

---

## 1. Comprendre le problème

On dispose d'un dataset de **1 460 maisons** avec leurs caractéristiques (surface, quartier, qualité, année de construction, etc.) et leur **prix de vente**. L'objectif est d'entraîner un modèle capable de prédire le prix de **1 459 nouvelles maisons** dont on ne connaît pas le prix.

Kaggle nous évalue sur la **MAE** (Mean Absolute Error) : la moyenne des écarts entre nos prédictions et les vrais prix. Si on prédit 200 000 $ et que la maison vaut 210 000 $, l'erreur est de 10 000 $. On veut minimiser cette moyenne.

---

## 2. Explorer les données

La première chose à faire, c'est comprendre **la variable qu'on essaie de prédire**. Voici la distribution du prix de vente :

![Distribution de SalePrice avant et après transformation log](/reports/figures/01_saleprice_distribution.png)

On voit clairement que la distribution brute (à gauche) est **asymétrique** : la majorité des maisons coûtent entre 100k et 250k$, avec une longue queue vers les prix élevés. Ce type de distribution pose problème car :
- Les valeurs extrêmes (maisons très chères) tirent les prédictions vers le haut
- Les modèles linéaires fonctionnent mieux avec des distributions symétriques

On applique donc une **transformation logarithmique** `log1p(x) = log(1 + x)`. Le résultat (à droite) est beaucoup plus symétrique, presque une courbe en cloche. C'est ce qu'on appelle une **distribution log-normale** — très courante pour les prix, les salaires, etc.

> On travaille en log pendant tout l'entraînement, puis on revient aux vrais dollars à la fin avec `expm1()`.

---

## 3. Supprimer les outliers

Avant d'entraîner quoi que ce soit, on vérifie s'il y a des **données aberrantes** qui pourraient fausser l'apprentissage.

![Détection des outliers : GrLivArea vs SalePrice](/reports/figures/02_outliers.png)

On repère **2 maisons** (les ✕ rouges) avec une surface > 4 000 sqft mais un prix anormalement bas (< 300k$). Ce sont probablement des ventes forcées (saisies bancaires, etc.). L'auteur du dataset recommande lui-même de les retirer.

Si on les garde, le modèle apprend que "grande surface = pas forcément cher", ce qui est faux dans 99% des cas. On les supprime donc du train set (mais on ne touche pas au test set, évidemment).

---

## 4. Nettoyer les données

Le dataset contient beaucoup de **valeurs manquantes** (NaN), mais elles n'ont pas toutes la même signification. On a lu la documentation du dataset pour comprendre chaque cas :

| Cas | Exemple | Ce que NaN signifie | Action |
|---|---|---|---|
| Pas d'équipement | `GarageType` = NaN | "Pas de garage" | → `"None"` |
| Surface inexistante | `GarageArea` = NaN | "0 sqft de garage" | → `0` |
| Erreur de saisie | `Electrical` = NaN | Vrai manquant | → Valeur la plus fréquente |
| Dépend du contexte | `LotFrontage` = NaN | Inconnu | → Médiane du quartier |

C'est ce qu'on appelle du **nettoyage domain-driven** : on utilise notre compréhension du domaine (l'immobilier) plutôt qu'une imputation automatique aveugle. Par exemple, remplacer `GarageArea = NaN` par la médiane globale serait une erreur — ça mettrait 400 sqft de garage à une maison qui n'en a pas !

On convertit aussi `MSSubClass` (un code de type de logement comme 20, 60, 120) en texte, car ce ne sont pas des quantités — "60" n'est pas "3 fois 20".

---

## 5. Encoder les variables ordinales

Beaucoup de colonnes contiennent du texte comme `"Excellent"`, `"Good"`, `"Average"`, `"Fair"`, `"Poor"`. Ce sont des variables **ordinales** — elles ont un ordre naturel.

Le problème : si on fait du One-Hot Encoding classique (une colonne par catégorie avec des 0/1), on **perd l'information d'ordre**. Le modèle ne sait pas que "Excellent" > "Good" > "Average".

Notre solution : on remplace chaque catégorie par un chiffre qui respecte l'ordre :

```python
{"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
```

On fait ça pour **15+ variables** : qualité extérieure, qualité cuisine, qualité garage, qualité sous-sol, finition du garage, type d'allée, climatisation, etc.

L'avantage pour les arbres de décision : ils peuvent maintenant faire des splits comme `ExterQual > 3` (= au-dessus de la moyenne), ce qui est beaucoup plus informatif que de tester chaque catégorie une par une.

---

## 6. Feature Engineering — Créer de nouvelles colonnes

C'est l'étape qui a eu **le plus d'impact** sur notre score. On crée de nouvelles colonnes à partir des colonnes existantes pour donner plus de signal au modèle.

### Surfaces agrégées
```python
TotalSF = Surface_sous-sol + Surface_RDC + Surface_étage
TotalBath = SDB_complètes + 0.5 × demi-SDB + SDB_sous-sol + ...
```
Un acheteur regarde la surface **totale**, pas chaque étage séparément.

### Âge au moment de la vente
```python
Age = Année_vente − Année_construction
```
L'année brute (1965) est moins informative que l'âge au moment de la vente (41 ans).

### Interactions Qualité × Surface ⭐
```python
QualSF = OverallQual × TotalSF
```
C'est notre **meilleure feature**. Le raisonnement : une grande maison de qualité moyenne et une petite maison de haute qualité se valent. Le produit capture cette interaction. On voit d'ailleurs dans la matrice de corrélation que `QualSF` est très fortement corrélé au prix :

![Matrice de corrélation entre les top features et SalePrice](/reports/figures/04_correlation_heatmap.png)

On observe que :
- `QualSF` (0.84), `OverallQual` (0.82), `GrLivArea` (0.72), et `TotalSF` (0.77) sont les features les plus corrélées au prix
- `Neighborhood_te` (notre target encoding) a aussi une forte corrélation (0.73)
- `Age` a une corrélation **négative** (−0.57) : plus la maison est vieille, moins elle vaut

---

## 7. Target Encoding — Transformer les quartiers en chiffres

La colonne `Neighborhood` contient 25 quartiers. Avec le One-Hot classique, ça crée 25 colonnes de 0/1 — c'est beaucoup de colonnes pour une seule information.

Notre approche : remplacer chaque quartier par le **prix moyen** (en log) des maisons de ce quartier. Ainsi `"StoneBr"` (quartier riche) → 12.45 et `"MeadowV"` (quartier modeste) → 11.55.

**Le piège du data leakage :** si on calcule la moyenne sur tout le train set, chaque maison "voit" indirectement son propre prix dans la moyenne de son quartier. C'est de la triche.

**Notre solution — OOF Target Encoding :** on divise le train en 5 groupes. Pour la maison du groupe 2, la moyenne du quartier est calculée **uniquement** sur les groupes 1, 3, 4, 5. Comme ça, la maison n'a jamais contribué au calcul de sa propre feature. Pas de leakage.

On applique cette technique à 4 variables : `Neighborhood`, `Condition1`, `Exterior1st`, `Exterior2nd`.

---

## 8. Réduire le bruit

### Supprimer les colonnes inutiles
On supprime les colonnes qui n'apportent aucun signal :
- `Utilities` : 99.9% de valeurs identiques ("AllPub")
- `Street` : 99.6% identiques ("Pave")
- `PoolQC` : 99% de NaN (quasi aucune maison n'a de piscine)
- `Id` : numéro de ligne, aucun lien avec le prix

### Corriger l'asymétrie (skewness)

Certaines colonnes numériques sont très asymétriques — comme `MiscVal` ou `LotArea` :

![Top 10 features les plus asymétriques avant correction](/reports/figures/03_skewness.png)

Les features en rouge dépassent notre seuil de skewness (0.75). On leur applique `log1p` pour les rendre plus symétriques, comme on l'a fait pour le prix. Les modèles apprennent mieux sur des features bien distribuées.

---

## 9. Nos 4 modèles de base

On n'utilise pas un seul modèle mais **quatre**, chacun avec ses forces :

### XGBoost
Un algorithme de **gradient boosting** qui construit des arbres de décision séquentiellement. Chaque arbre corrige les erreurs du précédent. C'est le modèle le plus populaire en compétition Kaggle.

Nos paramètres clés :
- `learning_rate = 0.008` : petits pas d'apprentissage pour une convergence fine
- `max_depth = 4` : arbres peu profonds pour éviter le surapprentissage
- `reg_lambda = 2.0` : régularisation L2 pour simplifier le modèle
- `early_stopping = 150` : on s'arrête si le modèle n'apprend plus

### CatBoost
Similaire à XGBoost mais spécialisé dans le traitement des **variables catégorielles**. Il les encode automatiquement via une technique interne (ordered target statistics). On l'optimise avec `loss_function="MAE"` pour cibler directement notre métrique Kaggle.

### ExtraTrees
Un ensemble d'arbres de décision qui choisissent leurs seuils de split **aléatoirement**. Ça semble contre-intuitif, mais cette randomisation crée une **diversité** complémentaire aux boosters.

### LightGBM
Variante de gradient boosting qui construit les arbres **feuille par feuille** (leaf-wise) au lieu de niveau par niveau (level-wise comme XGBoost). Cela capture potentiellement des patterns différents.

---

## 10. Courbe d'apprentissage — Vérifier qu'on ne surapprend pas

Le **surapprentissage** (overfitting), c'est quand le modèle "apprend par cœur" les données d'entraînement au lieu de comprendre les patterns. Il performe très bien sur le train mais mal sur de nouvelles données.

Pour le détecter, on trace la **courbe d'apprentissage** — l'erreur sur le train et sur la validation au fil des itérations :

![Courbe d'apprentissage XGBoost — Fold 1](/reports/figures/05_learning_curve_xgb.png)

On observe que :
- La courbe **train** (bleu) descend continuellement — le modèle fit de mieux en mieux les données d'entraînement
- La courbe **validation** (rouge) descend aussi, puis se stabilise ou remonte légèrement
- La ligne verte indique le point d'**early stopping** : le moment optimal où on arrête l'entraînement

Si on continuait au-delà, la courbe train continuerait à descendre mais la courbe validation remonterait — c'est l'overfitting. L'early stopping nous protège de ça en arrêtant au bon moment.

L'écart entre les deux courbes (le "gap") reste **raisonnable**, ce qui indique une bonne régularisation. Si le gap était énorme, ça signifierait un overfitting sévère.

---

## 11. Features les plus importantes

XGBoost nous fournit un classement d'importance : quelles features contribuent le plus aux prédictions ?

![Top 20 Features — XGBoost Feature Importance](/reports/figures/06_feature_importance.png)

Ce graphique confirme notre intuition :
- Les features **qu'on a créées** (`QualSF`, `TotalSF`, `QualFinishSF`, `OverallScore`) sont parmi les plus importantes — le feature engineering est payant
- `OverallQual` et `GrLivArea` restent des piliers (qualité × surface = le prix dans l'immobilier)
- Le **target encoding** (`Neighborhood_te`) ressort aussi — le quartier est un facteur clé
- Les variables d'âge (`Age`, `RemodAge`) ont un poids significatif

---

## 12. Prédictions Out-Of-Fold — Chaque modèle individuellement

Pour combiner les modèles (stacking), on a besoin de prédictions "honnêtes" de chaque modèle sur le train set. On utilise la **validation croisée** (K-Fold, K=5) : chaque maison est prédite par un modèle qui ne l'a **jamais vue** pendant son entraînement.

Voici les résultats de chaque modèle — le graphique montre les prédictions (axe Y) vs les vrais prix (axe X). Plus les points sont proches de la diagonale (ligne pointillée), mieux c'est :

![Prédictions OOF vs Prix Réel — Les 4 Modèles](/reports/figures/07_oof_scatter.png)

On constate que :
- **XGBoost** et **CatBoost** sont les plus fiables (points serrés autour de la diagonale)
- **ExtraTrees** est un peu plus dispersé mais capte certains patterns différemment
- **LightGBM** est très similaire à XGBoost (ce qui sera important plus tard)
- Tous les modèles ont plus de mal avec les maisons **très chères** (> 400k$) — c'est normal, il y a peu d'exemples

---

## 13. Le Stacking — Combiner les modèles

### Le principe
Au lieu de choisir le "meilleur" modèle, on les combine. Un **méta-modèle** apprend à pondérer les prédictions de chaque modèle base. C'est comme avoir un comité d'experts où chaque expert a un poids de vote différent.

### Ridge vs ElasticNet

On a testé deux méta-modèles :

![Comparaison des méta-modèles — Poids de chaque modèle](/reports/figures/08_meta_coefficients.png)

Le résultat est très intéressant :
- **Ridge** (à gauche) garde les 4 modèles avec des poids > 0
- **ElasticNet** (à droite) **élimine LightGBM** (coefficient = 0.000) !

Pourquoi ? L'ElasticNet inclut une régularisation **L1** (Lasso) qui peut mettre des coefficients à zéro. Il a détecté que LightGBM est **redondant** avec XGBoost — les deux sont des algorithmes de gradient boosting et font des erreurs très similaires. Garder les deux revient à compter deux fois le même avis.

L'ElasticNet a un MAE légèrement meilleur (13 768 vs 13 796 pour Ridge), on le garde.

**Poids finaux :**
| Modèle | Poids | Rôle |
|---|---|---|
| XGBoost | 0.577 | Pilier principal — le plus fiable |
| CatBoost | 0.342 | Complémentaire — capte les catégorielles |
| ExtraTrees | 0.102 | Diversité — apporte un signal orthogonal |
| LightGBM | 0.000 | Éliminé — redondant avec XGBoost |

---

## 14. Prédictions finales du stack

Voici les prédictions de notre modèle final (stack ElasticNet) :

![Stack ElasticNet : Prédictions vs Réel et Distribution des erreurs](/reports/figures/09_final_predictions.png)

**À gauche** : prédictions vs prix réels. Les points sont bien alignés sur la diagonale rouge (= prédiction parfaite). On voit quand même quelques écarts pour les maisons très chères (> 400k$).

**À droite** : la distribution des erreurs est centrée autour de 0 (pas de biais systématique) et la plupart des erreurs sont contenues entre −50k$ et +50k$. Le modèle ne sur-estime ni ne sous-estime de manière systématique.

---

## 15. Résultats finaux et progression

![Progression du score Kaggle sur 12 runs](/reports/figures/10_progression.png)

### Notre parcours en 12 itérations

| Run | Quoi de neuf | Score Kaggle | Ce qu'on apprend |
|---|---|---|---|
| 001 | Ridge simple (baseline) | 16 918 $ | Un point de départ |
| 003 | XGBoost + premières features | 14 125 $ | Le boosting est puissant |
| 008 | + Nettoyage avancé | 13 692 $ | Comprendre les données est crucial |
| 009 | Stacking 3 modèles | 13 375 $ | Combiner > choisir |
| 010 | + Encoding ordinal + LightGBM | 13 075 $ | L'ordre des catégories compte |
| 011 | + Target Encoding + skewness | 12 970 $ | Le quartier prédit le prix |
| **012** | **+ Tuning optimisé + ElasticNet** | **12 882 $** | **On approche de la limite** |

### Ce que j'ai retenu de ce projet

**1. Le feature engineering est le levier n°1.**
Créer `QualSF = Qualité × Surface` a eu plus d'impact que des heures de tuning d'hyperparamètres. En ML, on dit souvent *"garbage in, garbage out"* — mais l'inverse est vrai aussi : de bonnes features facilitent énormément le travail du modèle.

**2. Le stacking fonctionne parce que les modèles font des erreurs différentes.**
XGBoost, CatBoost et ExtraTrees ont des architectures différentes → ils se trompent sur des maisons différentes. En les combinant, les erreurs s'annulent partiellement.

**3. La régularisation est essentielle.**
Sans `early_stopping`, `reg_lambda`, et les garde-fous, les modèles surapprendraient et auraient de mauvais scores sur le test. La courbe d'apprentissage (section 10) le montre bien.

**4. Le nettoyage domain-driven fait la différence.**
Comprendre que `BsmtQual = NaN` signifie "pas de sous-sol" (et pas "donnée manquante") évite des erreurs d'imputation qui se propagent dans tout le pipeline.

**5. Les rendements sont décroissants.**
Les premiers runs ont gagné −2 800 $ d'erreur, les derniers seulement −88 $. C'est normal et attendu : il y a un plancher d'erreur irréductible lié au bruit naturel dans les données (ventes émotionnelles, négociations, circonstances personnelles). Aucun modèle ne pourra prédire ces facteurs humains.

**6. Le score CV n'est pas toujours corrélé au score Kaggle.**
Run 010 avait un meilleur CV que Run 011, mais un Kaggle pire. La validation croisée est un estimateur de la généralisation, pas une garantie. C'est pour ça qu'on soumet sur Kaggle pour valider.
