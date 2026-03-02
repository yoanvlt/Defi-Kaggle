"""
stacking.py — Fonctions partagées pour le stacking OOF.

Ce module centralise tout le code réutilisé entre les runs :
- Correction de skewness (train-aligned)
- Target Encoding OOF (anti-leakage)
- Génération de prédictions OOF universelle
- Construction de preprocessors (trees / linéaires)
- Évaluation et sélection du méta-modèle
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from catboost import Pool
import lightgbm as lgb


# ==========================================================
# Preprocessing
# ==========================================================

def build_tree_preprocessor(numeric_cols, categorical_cols):
    """Preprocesseur pour les modèles à base de trees (pas de scaling)."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])


def build_linear_preprocessor(numeric_cols, categorical_cols):
    """Preprocesseur pour les modèles linéaires (StandardScaler obligatoire)."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])


# ==========================================================
# Feature transforms
# ==========================================================

def fix_skewness(df, skewed_cols=None, threshold=0.75):
    """
    Correction de la skewness via log1p.
    
    Args:
        df: DataFrame
        skewed_cols: liste de colonnes à corriger (None = calculer depuis df)
        threshold: seuil de skewness
    
    Returns:
        (df_corrigé, liste_des_colonnes_skewed)
    
    Usage:
        X_train, skewed_cols = fix_skewness(X_train)          # calcul sur train
        X_test, _ = fix_skewness(X_test, skewed_cols=skewed_cols)  # applique au test
    """
    numeric_cols = df.select_dtypes(include=["int64", "int32", "float64"]).columns
    if skewed_cols is None:
        skewed_feats = df[numeric_cols].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats.abs() > threshold]
        skewed_cols = [col for col in skewed_feats.index if (df[col].dropna() >= 0).all()]
        print(f"  Skewness correction: {len(skewed_cols)} features")
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df, skewed_cols


def target_encode_oof(train_df, test_df, col, target, kf):
    """
    Target Encoding OOF (anti-leakage).
    
    Crée une colonne {col}_te dans train et test.
    """
    new_col = f"{col}_te"
    train_df[new_col] = np.nan
    global_mean = target.mean()
    for train_idx, val_idx in kf.split(train_df, target):
        means = target.iloc[train_idx].groupby(train_df[col].iloc[train_idx]).mean()
        train_df.loc[train_df.index[val_idx], new_col] = train_df[col].iloc[val_idx].map(means)
    train_df[new_col] = train_df[new_col].fillna(global_mean)
    full_means = target.groupby(train_df[col]).mean()
    test_df[new_col] = test_df[col].map(full_means).fillna(global_mean)
    return train_df, test_df


# ==========================================================
# OOF prediction generation
# ==========================================================

def get_oof_preds(model_name, model, X, y_log, X_test, kf, cat_features=None):
    """
    Génère des prédictions Out-of-Fold (OOF) pour stacking.
    
    Gère automatiquement :
    - XGBoost (Pipeline avec early stopping)
    - CatBoost (Pool + early stopping)
    - LightGBM (Pipeline avec early stopping)
    - Modèles linéaires (Pipeline avec preprocessor)
    - Tout autre modèle sklearn standard
    
    Returns:
        (oof_preds, mean_test_preds)
    """
    print(f"\n{'='*50}")
    print(f"Generating OOF for {model_name}...")
    oof_preds = np.zeros(len(X))
    test_preds_fold = []

    is_catboost = "CatBoost" in model_name
    is_xgb = "XGBoost" in model_name
    is_lgb = "LightGBM" in model_name
    is_linear = "Linear" in model_name

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

        if is_catboost:
            train_pool = Pool(X_train, y_train_log, cat_features=cat_features)
            val_pool = Pool(X_val, y_val_log, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)
        elif is_xgb:
            preprocessor = model.named_steps["preprocessor"]
            xgb_model = model.named_steps["xgb"]
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            xgb_model.fit(X_train_p, y_train_log,
                          eval_set=[(X_val_p, y_val_log)], verbose=False)
            val_pred_log = xgb_model.predict(X_val_p)
            test_pred_log = xgb_model.predict(X_test_p)
        elif is_lgb:
            preprocessor = model.named_steps["preprocessor"]
            lgb_model = model.named_steps["lgb"]
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            lgb_model.fit(X_train_p, y_train_log,
                          eval_set=[(X_val_p, y_val_log)],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
            val_pred_log = lgb_model.predict(X_val_p)
            test_pred_log = lgb_model.predict(X_test_p)
        elif is_linear:
            preprocessor = model.named_steps["preprocessor"]
            linear_model = model.named_steps["linear"]
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            linear_model.fit(X_train_p, y_train_log)
            val_pred_log = linear_model.predict(X_val_p)
            test_pred_log = linear_model.predict(X_test_p)
        else:
            model.fit(X_train, y_train_log)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)

        oof_preds[val_idx] = val_pred_log
        test_preds_fold.append(test_pred_log)
        fold_mae = mean_absolute_error(np.expm1(y_val_log), np.expm1(val_pred_log))
        print(f"  Fold {fold+1}: MAE = {fold_mae:.2f}")

    mean_test_preds = np.mean(test_preds_fold, axis=0)
    oof_mae = mean_absolute_error(np.expm1(y_log), np.expm1(oof_preds))
    print(f"  => {model_name} OOF MAE: {oof_mae:.2f}")
    return oof_preds, mean_test_preds


# ==========================================================
# Meta-model evaluation
# ==========================================================

def evaluate_meta(X_meta_train, y_log, X_meta_test, random_state=42):
    """
    Évalue Ridge vs ElasticNet comme méta-modèle et retourne le meilleur.
    
    Returns:
        dict avec keys: meta_model, meta_name, mae, mape, coefs,
                        final_pred (prédictions test en échelle originale)
    """
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta_train, y_log)
    ridge_pred = np.expm1(ridge.predict(X_meta_train))
    ridge_mae = mean_absolute_error(np.expm1(y_log), ridge_pred)

    # ElasticNet
    enet = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state)
    enet.fit(X_meta_train, y_log)
    enet_pred = np.expm1(enet.predict(X_meta_train))
    enet_mae = mean_absolute_error(np.expm1(y_log), enet_pred)

    print(f"Ridge      meta MAE: {ridge_mae:.2f} | Coefs: {dict(zip(X_meta_train.columns, ridge.coef_.round(4)))}")
    print(f"ElasticNet meta MAE: {enet_mae:.2f} | Coefs: {dict(zip(X_meta_train.columns, enet.coef_.round(4)))}")

    if enet_mae < ridge_mae:
        meta_model, meta_name, mae = enet, "ElasticNet", enet_mae
    else:
        meta_model, meta_name, mae = ridge, "Ridge", ridge_mae

    y_true = np.expm1(y_log)
    mape = mean_absolute_percentage_error(y_true, np.expm1(meta_model.predict(X_meta_train)))
    coefs = dict(zip(X_meta_train.columns, meta_model.coef_.round(4)))
    final_pred = np.expm1(meta_model.predict(X_meta_test))

    print(f"\n>>> Best Meta: {meta_name} (MAE={mae:.2f}, MAPE={mape*100:.2f}%)")
    print(f"    Coefs: {coefs}")

    return {
        "meta_model": meta_model,
        "meta_name": meta_name,
        "mae": mae,
        "mape": mape,
        "coefs": coefs,
        "final_pred": final_pred,
        "ridge_mae": ridge_mae,
        "enet_mae": enet_mae,
    }
