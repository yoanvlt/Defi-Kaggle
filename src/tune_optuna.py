"""
tune_optuna.py — Script d'optimisation des hyperparamètres pour Meta-Modèle ElasticNet

Ce script génère les prédictions (OOF) de tous les modèles de base
définis dans main.py, puis utilise Optuna pour trouver les meilleurs
hyperparamètres (alpha, l1_ratio) de l'ElasticNet de niveau 2.
"""

import sys
import os
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cleaning import remove_outliers
from src.stacking import build_tree_preprocessor, build_linear_preprocessor, fix_skewness, target_encode_oof, get_oof_preds
from src.models import AVAILABLE_MODELS, prepare_catboost_data
from main import CONFIG, run_preprocessing_pipeline

def objective(trial, oofs, y_log, kf_meta):
    # Les OOF ont déjà été générées une seule fois en amont
    X_meta = pd.DataFrame(oofs)
    
    # --- Hyperparamètres Optuna pour ElasticNet ---
    param = {
        "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": 10000,
        "random_state": 42
    }

    # --- Évaluation OOF Meta-Model ---
    oof_meta_preds = np.zeros(len(X_meta))
    for train_idx, val_idx in kf_meta.split(X_meta):
        X_tr, y_tr = X_meta.iloc[train_idx], y_log.iloc[train_idx]
        X_va, y_va = X_meta.iloc[val_idx], y_log.iloc[val_idx]

        meta_model = ElasticNet(**param)
        meta_model.fit(X_tr, y_tr)
        
        oof_meta_preds[val_idx] = meta_model.predict(X_va)

    # MAE sur les valeurs originales (exp)
    mae = mean_absolute_error(np.expm1(y_log), np.expm1(oof_meta_preds))
    return mae

if __name__ == "__main__":
    print("Début du Tuning Optuna pour Meta-Model ElasticNet !")
    print("Étape 1 : Chargement et préparation des données...")
    
    train_df = pd.read_csv(CONFIG["data"]["train_path"])
    train_df = remove_outliers(train_df)
    
    y_log = np.log1p(train_df["SalePrice"])
    X_train_raw = train_df.drop(columns=["SalePrice"])

    X = run_preprocessing_pipeline(X_train_raw, CONFIG["preprocessing_steps"])
    X_test_dummy = X.iloc[:2].copy() # Just for target encoder schema

    kf = KFold(n_splits=CONFIG['run_settings']['n_splits'], shuffle=True, random_state=42)
    for col in CONFIG["target_encode_cols"]:
        if col in X.columns:
            X, _ = target_encode_oof(X, X_test_dummy, col, y_log, kf)

    X = X.drop(columns=[c for c in CONFIG["drop_features"] if c in X.columns], errors="ignore")
    X, _ = fix_skewness(X)

    num_cols = X.select_dtypes(include=["int64", "int32", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    tree_pp = build_tree_preprocessor(num_cols, cat_cols)
    lin_pp = build_linear_preprocessor(num_cols, cat_cols)
    X_cat, _, cat_names = prepare_catboost_data(X.copy(), X_test_dummy.copy(), num_cols, cat_cols)

    print("Étape 2 : Génération des prédictions des Base Models (OOF)...")
    oofs = {}
    for model_name in CONFIG["active_models"]:
        print(f"  -> Calcul OOF pour {model_name}...")
        if "Linear" in model_name:
            model_instance = AVAILABLE_MODELS[model_name](lin_pp)
            data, cats = X, None
        elif "CatBoost" in model_name:
            model_instance = AVAILABLE_MODELS[model_name]()
            data, cats = X_cat, cat_names
        else:
            model_instance = AVAILABLE_MODELS[model_name](tree_pp)
            data, cats = X, None

        oof_p, _ = get_oof_preds(model_name, model_instance, data, y_log, X_test_dummy, kf, cat_features=cats)
        oofs[model_name] = oof_p
    
    print("\nÉtape 3 : Lancement de l'optimisation (100 trials) sur l'ElasticNet...")
    study = optuna.create_study(direction="minimize")
    # On passe nos OOF prè-calculées à l'objective via une fonction anonyme
    study.optimize(lambda trial: objective(trial, oofs, y_log, kf), n_trials=100)

    print("\nMeilleurs paramètres ElasticNet (Meta-Modèle) trouvés:")
    print(study.best_params)
    print(f"Meilleure MAE Meta-Modèle: {study.best_value:.4f}")
