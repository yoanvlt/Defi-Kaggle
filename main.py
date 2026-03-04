import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import KFold

# Allow absolute imports from project root
sys.path.append(os.path.dirname(__file__))

# Import de toutes nos méthodes
from src.cleaning import clean_data, remove_outliers
from src.features import add_features, encode_ordinals, add_poly_features
from src.advanced_features import add_advanced_features, add_spatial_clusters
from src.stacking import (
    build_tree_preprocessor, build_linear_preprocessor,
    fix_skewness, target_encode_oof, get_oof_preds, evaluate_meta,
)
from src.models import AVAILABLE_MODELS, prepare_catboost_data

# ==========================================================
# 1. CONFIGURATION DE L'EXPÉRIENCE
# ==========================================================
# Modifiez facilement cette section pour tester de nouvelles configurations

CONFIG = {
    "data": {
        "train_path": "Data/train.csv",
        "test_path": "Data/test.csv"
    },
    
    # 📌 pipeline de preprocessing complet: chaque fonction (df -> df) est appelée dans l'ordre
    "preprocessing_steps": [
        clean_data,             # Nettoyage de base (gestion des NaNs)
        encode_ordinals,        # Encodage manuel des variables qualitatives
        # add_spatial_clusters,   # Clustering spatial par quartier
        add_features,           # Features de base
        # add_poly_features,    # Features polynomiales (décommenté = activé)
        add_advanced_features   # Features métier (décommenté = activé)
    ],
    
    # 📌 Colonnes à dropper et encoder
    "drop_features": ["Utilities", "Street", "PoolQC", "PoolArea", "MiscFeature", "MiscVal", "Id", "HouseStyle_1.5Unf"],
    "target_encode_cols": ["Neighborhood", "Condition1", "Exterior1st", "Exterior2nd"],
    
    # 📌 Modèles à utiliser pour ce run (doivent exister dans AVAILABLE_MODELS dans src/models.py)
    "active_models": [
        "XGBoost",
        "CatBoost",
        "ExtraTrees",
        "LightGBM",
        "Linear_Ridge",
        "Linear_Lasso"
    ],
    
    "run_settings": {
        "random_state": 42,
        "n_splits": 5,
        "submission_path": "submissions/submission_run018.csv"
    }
}


# ==========================================================
# 2. PIPELINE DE PREPROCESSING CENTRALISÉ
# ==========================================================
def run_preprocessing_pipeline(df, pipeline_steps):
    """
    Applique séquentiellement toutes les fonctions de la liste 'pipeline_steps'.
    C'est ici qu'on gagne en modularité : ajoutez simplement une nouvelle fonction dans CONFIG!
    """
    df_out = df.copy()
    for step_func in pipeline_steps:
        df_out = step_func(df_out)
    return df_out


# ==========================================================
# 3. SCRIPT PRINCIPAL
# ==========================================================
def main():
    print(f"Lancement de la pipeline principale — {datetime.now()}")
    settings = CONFIG["run_settings"]

    # --- Étape A : Chargement et Nettoyage Train/Test ---
    train_df = pd.read_csv(CONFIG["data"]["train_path"])
    train_df = remove_outliers(train_df)
    
    test_df = pd.read_csv(CONFIG["data"]["test_path"])
    test_ids = test_df["Id"].copy()

    y_log = np.log1p(train_df["SalePrice"])
    X_train_raw = train_df.drop(columns=["SalePrice"])
    X_test_raw = test_df

    # --- Étape B : Pipeline métier ---
    X = run_preprocessing_pipeline(X_train_raw, CONFIG["preprocessing_steps"])
    X_test = run_preprocessing_pipeline(X_test_raw, CONFIG["preprocessing_steps"])

    # --- Étape C : Target Encoding (OOF) ---
    kf = KFold(n_splits=settings["n_splits"], shuffle=True, random_state=settings["random_state"])
    for col in CONFIG["target_encode_cols"]:
        if col in X.columns:
            X, X_test = target_encode_oof(X, X_test, col, y_log, kf)

    # --- Étape D : Nettoyage final et Skewness ---
    X = X.drop(columns=[c for c in CONFIG["drop_features"] if c in X.columns], errors="ignore")
    X_test = X_test.drop(columns=[c for c in CONFIG["drop_features"] if c in X_test.columns], errors="ignore")
    
    X, skewed = fix_skewness(X)
    X_test, _ = fix_skewness(X_test, skewed_cols=skewed)

    # --- Étape E : Préparation Scikit-learn (Preprocessors) ---
    num_cols = X.select_dtypes(include=["int64", "int32", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    tree_pp = build_tree_preprocessor(num_cols, cat_cols)
    lin_pp = build_linear_preprocessor(num_cols, cat_cols)
    print(f"Colonnes utilisées prêtes : {len(num_cols)} num, {len(cat_cols)} cat")

    X_cat, X_test_cat, cat_names = prepare_catboost_data(X, X_test, num_cols, cat_cols)

    # --- Étape F : Génération des prédictions (Base Models) ---
    oofs, tests = {}, {}
    for model_name in CONFIG["active_models"]:
        if model_name not in AVAILABLE_MODELS:
            print(f"Attention: le modèle {model_name} n'existe pas dans le registre ! Ignoré.")
            continue
            
        print(f"\n--- Entraînement de {model_name} ---")
        
        # Récupération de l'instance via le dictionnaire global
        # On passe preprocessor selon le type de modèle (lié au nom ou défini dans le registre)
        if "Linear" in model_name:
            model_instance = AVAILABLE_MODELS[model_name](lin_pp)
            data, test_data, cats = X, X_test, None
        elif "CatBoost" in model_name:
            # CatBoost n'a pas besoin de preprocessor explicite via sklearn
            model_instance = AVAILABLE_MODELS[model_name]()
            data, test_data, cats = X_cat, X_test_cat, cat_names
        else:
            model_instance = AVAILABLE_MODELS[model_name](tree_pp)
            data, test_data, cats = X, X_test, None

        oof_p, test_p = get_oof_preds(model_name, model_instance, data, y_log, test_data, kf, cat_features=cats)
        oofs[model_name], tests[model_name] = oof_p, test_p

    # --- Étape G : Méta-Modèle (Stacking) ---
    print("\n--- Évaluation du Méta-Modèle ---")
    X_meta = pd.DataFrame(oofs)
    X_meta_test = pd.DataFrame(tests)
    result = evaluate_meta(X_meta, y_log, X_meta_test)

    # --- Étape H : Sauvegarde des résultats ---
    print(f"\n*** RUN PRINCIPAL — MAE: {result['mae']:.4f}, MAPE: {result['mape']*100:.2f}% ***")
    
    sub = pd.DataFrame({"Id": test_ids, "SalePrice": result["final_pred"]})
    os.makedirs(os.path.dirname(settings["submission_path"]), exist_ok=True)
    sub.to_csv(settings["submission_path"], index=False)
    print(f"Fichier de soumission créé : {settings['submission_path']}")

if __name__ == "__main__":
    main()
