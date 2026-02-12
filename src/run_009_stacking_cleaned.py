import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_features
from src.cleaning import clean_data, remove_outliers

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_009_stacking_cleaned.csv"
RESULTS_PATH = "results_run009.txt"
RANDOM_STATE = 42
N_SPLITS = 5

def get_oof_preds(model_name, model, X, y_log, X_test, kf, cat_features=None):
    """
    Génère les prédictions OOF (Out-Of-Fold) pour le train et la moyenne des prédictions pour le test.
    """
    print(f"Generating OOF for {model_name}...")
    oof_preds = np.zeros(len(X))
    test_preds_fold = []
    
    is_catboost = "CatBoost" in model_name
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        if is_catboost:
            train_pool = Pool(X_train, y_train_log, cat_features=cat_features)
            val_pool = Pool(X_val, y_val_log, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)
        else:
            model.fit(X_train, y_train_log)
            val_pred_log = model.predict(X_val)
            test_pred_log = model.predict(X_test)
            
        oof_preds[val_idx] = val_pred_log
        test_preds_fold.append(test_pred_log)
        
    mean_test_preds = np.mean(test_preds_fold, axis=0)
    return oof_preds, mean_test_preds

def main():
    print(f"Run 009: Stacking + Advanced Cleaning started at {datetime.now()}")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. Outlier Removal (TRAIN ONLY)
    print("Removing outliers from Train set...")
    original_len = len(train_df)
    train_df = remove_outliers(train_df)
    print(f"Train set size: {original_len} -> {len(train_df)}")

    # 3. Global Preprocessing (Clean + FE)
    def preprocess_global(df):
        df_clean = clean_data(df)
        df_feat = add_features(df_clean)
        return df_feat
    
    print("Applying Global Cleaning & Feature Engineering...")
    X_all = preprocess_global(train_df.drop(columns=["SalePrice"]))
    y_log = np.log1p(train_df["SalePrice"])
    X_test_all = preprocess_global(test_df)
    
    # 4. Prepare Models
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    numeric_cols = X_all.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_all.select_dtypes(include=["object"]).columns
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    # Preprocessor for Sklearn (XGB, ET)
    # Note: Imputers are still good safety nets even if data is "clean"
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])
    
    # A. XGBoost (Tuned)
    model_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBRegressor(
            n_estimators=4000, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.75, 
            colsample_bytree=0.75,
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ))
    ])
    
    # B. CatBoost (Native)
    # Needs columns to be string/filled
    X_cat = X_all.copy()
    X_test_cat = X_test_all.copy()
    cat_features_indices = []
    for col in categorical_cols:
        X_cat[col] = X_cat[col].astype(str).fillna("Missing")
        X_test_cat[col] = X_test_cat[col].astype(str).fillna("Missing")
        cat_features_indices.append(col)
        
    model_cat = CatBoostRegressor(
        loss_function="MAE",
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        silent=True
    )
    
    # C. ExtraTrees
    model_et = Pipeline([
        ("preprocessor", preprocessor),
        ("et", ExtraTreesRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    # 5. Generate OOFs
    oof_xgb, test_xgb = get_oof_preds("XGBoost", model_xgb, X_all, y_log, X_test_all, kf)
    oof_cat, test_cat = get_oof_preds("CatBoost", model_cat, X_cat, y_log, X_test_cat, kf, cat_features=cat_features_indices)
    oof_et, test_et = get_oof_preds("ExtraTrees", model_et, X_all, y_log, X_test_all, kf)
    
    # 6. Stacking (Meta Model)
    X_meta_train = pd.DataFrame({"XGB": oof_xgb, "Cat": oof_cat, "ET": oof_et})
    X_meta_test = pd.DataFrame({"XGB": test_xgb, "Cat": test_cat, "ET": test_et})
    
    print("Training Meta-Model (Ridge)...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_log)
    
    print(f"Meta Coefficients: {meta_model.coef_}")
    
    # Check OOF Performance
    meta_oof_pred = np.expm1(meta_model.predict(X_meta_train))
    y_true = np.expm1(y_log)
    mae = mean_absolute_error(y_true, meta_oof_pred)
    mape = mean_absolute_percentage_error(y_true, meta_oof_pred)
    
    print(f"Stacking OOF MAE: {mae:.4f}")
    print(f"Stacking OOF MAPE: {mape*100:.2f}%")
    
    # 7. Submission
    final_pred = np.expm1(meta_model.predict(X_meta_test))
    
    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": final_pred
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Stacking Cleaned MAE: {mae:.4f}\n")
        f.write(f"Stacking Cleaned MAPE: {mape:.4f}\n")
        f.write(f"Meta Coefs: {meta_model.coef_}\n")

if __name__ == "__main__":
    main()
