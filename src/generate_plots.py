"""
Generates all analysis plots for the RUN 012 walkthrough.
Produces: distribution plots, learning curves, feature importance,
scatter plots, bar charts, etc.
"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import skew
from sklearn.model_selection import KFold, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_features, encode_ordinals
from src.cleaning import clean_data, remove_outliers

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

TRAIN_PATH = "Data/train.csv"
OUT_DIR = "reports/figures"
RANDOM_STATE = 42
N_SPLITS = 5
SKEW_THRESHOLD = 0.75
TARGET_ENCODE_COLS = ["Neighborhood", "Condition1", "Exterior1st", "Exterior2nd"]
DROP_FEATURES = ["Utilities", "Street", "PoolQC", "PoolArea", "MiscFeature", "MiscVal", "Id"]

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load and basic prep
# ──────────────────────────────────────────────
train_raw = pd.read_csv(TRAIN_PATH)

# PLOT 1: SalePrice distribution before/after log
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(train_raw["SalePrice"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_title("Distribution de SalePrice (brut)", fontweight="bold")
axes[0].set_xlabel("Prix ($)")
axes[0].set_ylabel("Nombre de maisons")
axes[0].axvline(train_raw["SalePrice"].median(), color="red", ls="--", label=f'Médiane: {train_raw["SalePrice"].median():,.0f}$')
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

y_log = np.log1p(train_raw["SalePrice"])
axes[1].hist(y_log, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
axes[1].set_title("Distribution de log1p(SalePrice)", fontweight="bold")
axes[1].set_xlabel("log(1 + Prix)")
axes[1].set_ylabel("Nombre de maisons")
axes[1].axvline(y_log.median(), color="red", ls="--", label=f'Médiane: {y_log.median():.2f}')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_saleprice_distribution.png")
plt.close()
print("✅ 01 SalePrice distribution")

# PLOT 2: Outliers
fig, ax = plt.subplots(figsize=(8, 5))
outlier_mask = (train_raw["GrLivArea"] > 4000) & (train_raw["SalePrice"] < 300000)
normal = train_raw[~outlier_mask]
outliers = train_raw[outlier_mask]
ax.scatter(normal["GrLivArea"], normal["SalePrice"], alpha=0.4, s=15, c="#4C72B0", label="Normal")
ax.scatter(outliers["GrLivArea"], outliers["SalePrice"], c="red", s=80, marker="X", zorder=5, label=f"Outliers ({len(outliers)})")
ax.set_xlabel("Surface habitable (GrLivArea, sqft)")
ax.set_ylabel("Prix de vente ($)")
ax.set_title("Détection des Outliers : GrLivArea vs SalePrice", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
ax.legend()
plt.savefig(f"{OUT_DIR}/02_outliers.png")
plt.close()
print("✅ 02 Outliers")

# ──────────────────────────────────────────────
# 2. Clean + FE
# ──────────────────────────────────────────────
train_df = remove_outliers(train_raw)
y = train_df["SalePrice"]
y_log = np.log1p(y)
X_raw = train_df.drop(columns=["SalePrice"])
X_all = add_features(encode_ordinals(clean_data(X_raw)))

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Target encoding OOF
for col in TARGET_ENCODE_COLS:
    if col in X_all.columns:
        new_col = f"{col}_te"
        X_all[new_col] = np.nan
        global_mean = y_log.mean()
        for train_idx, val_idx in kf.split(X_all, y_log):
            means = y_log.iloc[train_idx].groupby(X_all[col].iloc[train_idx]).mean()
            X_all.loc[X_all.index[val_idx], new_col] = X_all[col].iloc[val_idx].map(means)
        X_all[new_col] = X_all[new_col].fillna(global_mean)

# PLOT 3: Skewness before/after
numeric_before = X_all.select_dtypes(include=["int64","int32","float64"]).columns
skew_before = X_all[numeric_before].apply(lambda x: skew(x.dropna()))
top_skewed = skew_before.abs().nlargest(10)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#C44E52" if v > 0.75 else "#55A868" for v in top_skewed.values]
bars = ax.barh(range(len(top_skewed)), top_skewed.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(top_skewed)))
ax.set_yticklabels(top_skewed.index)
ax.set_xlabel("Skewness (asymétrie)")
ax.set_title("Top 10 Features les plus asymétriques (avant correction)", fontweight="bold")
ax.axvline(0.75, color="gray", ls="--", alpha=0.7, label="Seuil = 0.75")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_skewness.png")
plt.close()
print("✅ 03 Skewness")

# Apply skewness fix
cols_to_drop = [c for c in DROP_FEATURES if c in X_all.columns]
X_all = X_all.drop(columns=cols_to_drop, errors="ignore")

numeric_cols_all = X_all.select_dtypes(include=["int64","int32","float64"]).columns
skewed_feats = X_all[numeric_cols_all].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats.abs() > SKEW_THRESHOLD]
for col in skewed_feats.index:
    if (X_all[col].dropna() >= 0).all():
        X_all[col] = np.log1p(X_all[col])

# ──────────────────────────────────────────────
# 3. Prepare models  
# ──────────────────────────────────────────────
numeric_cols = X_all.select_dtypes(include=["int64","int32","float64"]).columns
categorical_cols = X_all.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])

# PLOT 4: Correlation heatmap of top features  
top_feats = ["OverallQual","GrLivArea","TotalSF","QualSF","GarageCars","TotalBath",
             "ExterQual","KitchenQual","Neighborhood_te","Age"]
avail_feats = [f for f in top_feats if f in X_all.columns]
corr_df = X_all[avail_feats].copy()
corr_df["SalePrice_log"] = y_log.values
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr.iloc[i,j]) > 0.6 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matrice de corrélation — Top Features vs SalePrice", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png")
plt.close()
print("✅ 04 Correlation heatmap")

# ──────────────────────────────────────────────
# 4. Train models with tracking for learning curves
# ──────────────────────────────────────────────

# XGBoost with eval tracking
print("\n🔄 Training XGBoost with eval history...")
X_all_processed_list = []
fold_train_curves = []
fold_val_curves = []
oof_xgb = np.zeros(len(X_all))
oof_cat = np.zeros(len(X_all))
oof_et = np.zeros(len(X_all))
oof_lgb = np.zeros(len(X_all))

# Catboost prep
X_cat = X_all.copy()
cat_feats = []
for col in categorical_cols:
    X_cat[col] = X_cat[col].astype(str).fillna("Missing")
    cat_feats.append(col)
for col in numeric_cols:
    X_cat[col] = X_cat[col].fillna(0)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
xgb_importances = np.zeros(0)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_log)):
    X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    # XGBoost
    pre = preprocessor.__class__(preprocessor.transformers)
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    
    xgb_m = XGBRegressor(
        n_estimators=3000, learning_rate=0.008, max_depth=4,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0,
        min_child_weight=3, gamma=0.01, early_stopping_rounds=150,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mae"
    )
    xgb_m.fit(X_tr_p, y_tr, eval_set=[(X_tr_p, y_tr), (X_va_p, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_m.predict(X_va_p)
    
    if fold == 0:
        results = xgb_m.evals_result()
        fold_train_curves = results["validation_0"]["mae"]
        fold_val_curves = results["validation_1"]["mae"]
        xgb_importances = xgb_m.feature_importances_
    
    # CatBoost
    X_tr_cat, X_va_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    cat_m = CatBoostRegressor(
        loss_function="MAE", iterations=3000, learning_rate=0.02,
        depth=6, l2_leaf_reg=5, bagging_temperature=0.5, random_strength=0.5,
        random_seed=RANDOM_STATE, allow_writing_files=False, silent=True,
        early_stopping_rounds=100
    )
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_feats)
    val_pool = Pool(X_va_cat, y_va, cat_features=cat_feats)
    cat_m.fit(train_pool, eval_set=val_pool, verbose=False)
    oof_cat[val_idx] = cat_m.predict(X_va_cat)
    
    # ExtraTrees
    pre2 = preprocessor.__class__(preprocessor.transformers)
    X_tr_p2 = pre2.fit_transform(X_tr)
    X_va_p2 = pre2.transform(X_va)
    et_m = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_split=3,
                                min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    et_m.fit(X_tr_p2, y_tr)
    oof_et[val_idx] = et_m.predict(X_va_p2)
    
    # LightGBM
    pre3 = preprocessor.__class__(preprocessor.transformers)
    X_tr_p3 = pre3.fit_transform(X_tr)
    X_va_p3 = pre3.transform(X_va)
    lgb_m = LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, max_depth=4, num_leaves=20,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=2.0,
        min_child_samples=10, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_m.fit(X_tr_p3, y_tr, eval_set=[(X_va_p3, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = lgb_m.predict(X_va_p3)
    
    print(f"  Fold {fold+1} done")

print("✅ All folds trained")

# PLOT 5: Learning curves (XGBoost fold 1)
fig, ax = plt.subplots(figsize=(10, 5))
n_iters = len(fold_train_curves)
x_range = range(0, n_iters, max(1, n_iters//200))
train_sub = [fold_train_curves[i] for i in x_range]
val_sub = [fold_val_curves[i] for i in x_range]
ax.plot(list(x_range), train_sub, label="Train MAE", color="#4C72B0", alpha=0.8)
ax.plot(list(x_range), val_sub, label="Validation MAE", color="#C44E52", alpha=0.8)
best_iter = xgb_m.best_iteration if hasattr(xgb_m, 'best_iteration') else np.argmin(fold_val_curves)
ax.axvline(best_iter, color="green", ls="--", alpha=0.7, label=f"Best iteration: {best_iter}")
ax.set_xlabel("Nombre d'arbres (itérations)")
ax.set_ylabel("MAE (log-scale)")
ax.set_title("Courbe d'apprentissage XGBoost — Fold 1", fontweight="bold")
ax.legend()
# Add annotation
ax.annotate("Zone d'overfitting\n(train ↓ mais val ↑)", 
            xy=(best_iter + 200, fold_val_curves[min(best_iter+200, n_iters-1)]),
            fontsize=9, color="#C44E52", ha="left",
            arrowprops=dict(arrowstyle="->", color="#C44E52"))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_learning_curve_xgb.png")
plt.close()
print("✅ 05 Learning curve XGBoost")

# PLOT 6: Feature importance (XGBoost)
if len(xgb_importances) > 0:
    # Get feature names after preprocessing
    pre_final = preprocessor.__class__(preprocessor.transformers)
    pre_final.fit(X_all)
    feat_names = []
    for name, trans, cols in pre_final.transformers_:
        if name == "num":
            feat_names.extend(cols)
        elif name == "cat":
            encoder = trans.named_steps["encoder"]
            feat_names.extend(encoder.get_feature_names_out(cols))
    
    if len(feat_names) == len(xgb_importances):
        imp_df = pd.DataFrame({"feature": feat_names, "importance": xgb_importances})
    else:
        imp_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(xgb_importances))], "importance": xgb_importances})
    
    top20 = imp_df.nlargest(20, "importance")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top20)), top20["importance"].values, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 20 Features — XGBoost Feature Importance", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_feature_importance.png")
    plt.close()
    print("✅ 06 Feature importance")

# PLOT 7: OOF predictions vs actual (scatter)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
models_oof = [("XGBoost", oof_xgb), ("CatBoost", oof_cat), ("ExtraTrees", oof_et), ("LightGBM", oof_lgb)]
colors_m = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (name, oof) in enumerate(models_oof):
    ax = axes[i]
    actual = np.expm1(y_log)
    predicted = np.expm1(oof)
    mae = mean_absolute_error(actual, predicted)
    ax.scatter(actual, predicted, alpha=0.3, s=10, c=colors_m[i])
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', alpha=0.5, label="Parfait")
    ax.set_title(f"{name}\nMAE: {mae:,.0f}$", fontweight="bold", fontsize=10)
    ax.set_xlabel("Prix réel ($)" if i == 0 else "")
    ax.set_ylabel("Prix prédit ($)" if i == 0 else "")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.legend(fontsize=8)
plt.suptitle("Prédictions OOF vs Prix Réel — Les 4 Modèles", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_oof_scatter.png")
plt.close()
print("✅ 07 OOF scatter")

# ──────────────────────────────────────────────
# 5. Meta-model
# ──────────────────────────────────────────────
X_meta = pd.DataFrame({"XGBoost": oof_xgb, "CatBoost": oof_cat, "ExtraTrees": oof_et, "LightGBM": oof_lgb})

ridge = Ridge(alpha=1.0)
ridge.fit(X_meta, y_log)
enet = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE)
enet.fit(X_meta, y_log)

# PLOT 8: Meta-model coefficients comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_names = X_meta.columns.tolist()
ridge_coefs = ridge.coef_
enet_coefs = enet.coef_

bars1 = axes[0].bar(model_names, ridge_coefs, color=["#4C72B0","#DD8452","#55A868","#C44E52"], edgecolor="white")
axes[0].set_title("Ridge (L2)\nGarde tous les modèles", fontweight="bold")
axes[0].set_ylabel("Coefficient (poids)")
for bar, val in zip(bars1, ridge_coefs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=10)
axes[0].set_ylim(0, max(ridge_coefs)*1.3)

bars2 = axes[1].bar(model_names, enet_coefs, color=["#4C72B0","#DD8452","#55A868","#C44E52"], edgecolor="white")
axes[1].set_title("ElasticNet (L1+L2)\nÉlimine LightGBM !", fontweight="bold")
axes[1].set_ylabel("Coefficient (poids)")
for bar, val in zip(bars2, enet_coefs):
    axes[1].text(bar.get_x() + bar.get_width()/2, max(val + 0.01, 0.01), f'{val:.3f}', ha='center', fontsize=10)
axes[1].set_ylim(0, max(enet_coefs)*1.3 if max(enet_coefs) > 0 else 1)
plt.suptitle("Comparaison des Méta-Modèles — Poids de chaque modèle", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_meta_coefficients.png")
plt.close()
print("✅ 08 Meta coefficients")

# PLOT 9: Final stacked prediction vs actual
meta_pred = np.expm1(enet.predict(X_meta))
actual = np.expm1(y_log)
residuals = meta_pred - actual

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Scatter
axes[0].scatter(actual, meta_pred, alpha=0.3, s=12, c="#4C72B0")
axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', alpha=0.7, label="Prédiction parfaite")
axes[0].set_xlabel("Prix réel ($)")
axes[0].set_ylabel("Prix prédit (Stack) ($)")
axes[0].set_title("Stack ElasticNet: Prédit vs Réel", fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[0].legend()

# Residuals
axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", ls="--")
axes[1].set_xlabel("Erreur (Prédit − Réel) en $")
axes[1].set_ylabel("Nombre de maisons")
axes[1].set_title(f"Distribution des erreurs\nMAE = {np.mean(np.abs(residuals)):,.0f}$", fontweight="bold")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_final_predictions.png")
plt.close()
print("✅ 09 Final predictions")

# PLOT 10: Run progression bar chart
runs = ["001\nBaseline", "003\nXGB+FE", "008\nCleaning", "009\nStack", "010\n+Ordinals", "011\n+TE", "012\n+Tuning"]
scores = [16918, 14125, 13692, 13375, 13075, 12970, 12882]
colors_r = ["#aaa"] * 5 + ["#DD8452", "#4C72B0"]
colors_r = ["#bbb", "#aaa", "#999", "#888", "#777", "#DD8452", "#4C72B0"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(runs, scores, color=colors_r, edgecolor="white", width=0.65)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{score:,}$', 
            ha='center', fontsize=10, fontweight="bold")
ax.set_ylabel("Score Kaggle (MAE en $)")
ax.set_title("Progression du score Kaggle sur 12 runs", fontweight="bold")
ax.set_ylim(12000, 17500)
# Arrow showing improvement
ax.annotate(f'−{16918-12882:,}$ (−{(16918-12882)/16918*100:.1f}%)', 
            xy=(6, 12882), xytext=(4.5, 16500),
            fontsize=12, fontweight="bold", color="#4C72B0",
            arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=2))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_progression.png")
plt.close()
print("✅ 10 Progression")

print("\n🎉 All 10 plots saved to reports/figures/")
