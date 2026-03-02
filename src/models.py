"""
models.py — Factory de modèles pré-configurés.

Chaque fonction retourne un modèle (ou Pipeline) prêt à l'emploi.
Les hyperparamètres correspondent au meilleur tuning (RUN 012/013).
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


RANDOM_STATE = 42


def get_xgboost(preprocessor):
    """XGBoost tunée (RUN 012+)."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBRegressor(
            n_estimators=8000, learning_rate=0.008, max_depth=4,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=2.0,
            min_child_weight=3, gamma=0.01,
            early_stopping_rounds=150,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])


def get_catboost():
    """CatBoost — gère les catégories nativement (pas de preprocessor)."""
    return CatBoostRegressor(
        loss_function="MAE", iterations=6000, learning_rate=0.02,
        depth=6, l2_leaf_reg=5, bagging_temperature=0.5,
        random_strength=0.5, random_seed=RANDOM_STATE,
        allow_writing_files=False, silent=True
    )


def get_extratrees(preprocessor):
    """ExtraTrees — forte randomisation, bon diversifieur."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("et", ExtraTreesRegressor(
            n_estimators=800, max_depth=20,
            min_samples_split=3, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])


def get_lgbm(preprocessor):
    """LightGBM — complémentaire à XGBoost."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("lgb", LGBMRegressor(
            n_estimators=8000, learning_rate=0.008, max_depth=4,
            num_leaves=20, subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.2, reg_lambda=2.0, min_child_samples=10,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ))
    ])


def get_ridge(preprocessor):
    """Ridge — modèle linéaire L2 (nécessite StandardScaler)."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("linear", Ridge(alpha=10.0, random_state=RANDOM_STATE))
    ])


def get_lasso(preprocessor):
    """Lasso — modèle linéaire L1 (sélection de features, nécessite StandardScaler)."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("linear", Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE))
    ])


def prepare_catboost_data(X, X_test, numeric_cols, categorical_cols):
    """
    Prépare les données pour CatBoost (conversion str + fillna).
    
    Returns:
        (X_cat, X_test_cat, cat_feature_names)
    """
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    cat_names = []
    for col in categorical_cols:
        X_cat[col] = X_cat[col].astype(str).fillna("Missing")
        X_test_cat[col] = X_test_cat[col].astype(str).fillna("Missing")
        cat_names.append(col)
    for col in numeric_cols:
        X_cat[col] = X_cat[col].fillna(0)
        X_test_cat[col] = X_test_cat[col].fillna(0)
    return X_cat, X_test_cat, cat_names

# ==========================================================
# REGISTRE DES MODÈLES (POUR MODULARITÉ)
# ==========================================================
# Pour ajouter un nouveau modèle, créez sa fonction plus haut et ajoutez-la ici.
AVAILABLE_MODELS = {
    "XGBoost": get_xgboost,
    "CatBoost": get_catboost,
    "ExtraTrees": get_extratrees,
    "LightGBM": get_lgbm,
    "Linear_Ridge": get_ridge,
    "Linear_Lasso": get_lasso
}
