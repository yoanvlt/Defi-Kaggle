import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from scipy.stats import skew
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
import lightgbm as lgb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_features, encode_ordinals
from src.cleaning import clean_data, remove_outliers

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_012_tuned_stack.csv"
RESULTS_PATH = "results_run012.txt"
RANDOM_STATE = 42
N_SPLITS = 5

# Features to drop
DROP_FEATURES = ["Utilities", "Street", "PoolQC", "PoolArea", "MiscFeature", "MiscVal", "Id"]

# Target encoding columns
TARGET_ENCODE_COLS = ["Neighborhood", "Condition1", "Exterior1st", "Exterior2nd"]

SKEW_THRESHOLD = 0.75


def fix_skewness(df, threshold=SKEW_THRESHOLD):
    numeric_cols = df.select_dtypes(include=["int64", "int32", "float64"]).columns
    skewed_feats = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats.abs() > threshold]
    print(f"  Skewness correction: {len(skewed_feats)} features (threshold={threshold})")
    for col in skewed_feats.index:
        if (df[col].dropna() >= 0).all():
            df[col] = np.log1p(df[col])
    return df


def target_encode_oof(train_df, test_df, col, target, kf):
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


def get_oof_preds(model_name, model, X, y_log, X_test, kf, cat_features=None):
    print(f"\n{'='*50}")
    print(f"Generating OOF for {model_name}...")
    oof_preds = np.zeros(len(X))
    test_preds_fold = []
    
    is_catboost = "CatBoost" in model_name
    is_xgb = "XGBoost" in model_name
    is_lgb = "LightGBM" in model_name
    
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
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            xgb_model.fit(
                X_train_processed, y_train_log,
                eval_set=[(X_val_processed, y_val_log)],
                verbose=False
            )
            val_pred_log = xgb_model.predict(X_val_processed)
            test_pred_log = xgb_model.predict(X_test_processed)
        elif is_lgb:
            preprocessor = model.named_steps["preprocessor"]
            lgb_model = model.named_steps["lgb"]
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            lgb_model.fit(
                X_train_processed, y_train_log,
                eval_set=[(X_val_processed, y_val_log)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            val_pred_log = lgb_model.predict(X_val_processed)
            test_pred_log = lgb_model.predict(X_test_processed)
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


def main():
    print(f"Run 012: Tuned Stacking + ElasticNet Meta started at {datetime.now()}")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df["Id"].copy()
    
    # 2. Outlier Removal (TRAIN ONLY)
    print("Removing outliers...")
    original_len = len(train_df)
    train_df = remove_outliers(train_df)
    print(f"Train: {original_len} -> {len(train_df)}")

    # 3. Separate target
    y = train_df["SalePrice"]
    y_log = np.log1p(y)
    X_train_raw = train_df.drop(columns=["SalePrice"])
    X_test_raw = test_df.copy()
    
    # 4. Clean + Ordinals + FE
    def preprocess_global(df):
        return add_features(encode_ordinals(clean_data(df)))
    
    print("Preprocessing...")
    X_all = preprocess_global(X_train_raw)
    X_test_all = preprocess_global(X_test_raw)
    
    # 5. Target Encoding OOF
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print("\nTarget Encoding OOF...")
    for col in TARGET_ENCODE_COLS:
        if col in X_all.columns:
            X_all, X_test_all = target_encode_oof(X_all, X_test_all, col, y_log, kf)
            print(f"  {col} -> {col}_te")
    
    # 6. Drop noise features
    cols_to_drop = [c for c in DROP_FEATURES if c in X_all.columns]
    X_all = X_all.drop(columns=cols_to_drop, errors="ignore")
    X_test_all = X_test_all.drop(columns=cols_to_drop, errors="ignore")
    
    # 7. Fix Skewness
    print("\nFixing skewness...")
    X_all = fix_skewness(X_all)
    X_test_all = fix_skewness(X_test_all)
    
    # 8. Prepare Models — TUNED HYPERPARAMETERS
    numeric_cols = X_all.select_dtypes(include=["int64", "int32", "float64"]).columns
    categorical_cols = X_all.select_dtypes(include=["object"]).columns
    print(f"\nFeatures: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])
    
    # A. XGBoost — TUNED: lower lr, more trees, deeper
    model_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBRegressor(
            n_estimators=8000,            # More trees (ES will cut)
            learning_rate=0.008,          # Lower LR for finer convergence
            max_depth=4,                  # Slightly shallower to reduce overfitting
            subsample=0.7,               
            colsample_bytree=0.6,         # Less features per tree = more diversity
            reg_alpha=0.1,                # Stronger L1
            reg_lambda=2.0,               # Stronger L2
            min_child_weight=3,           # NEW: prevent overfitting on small leaves
            gamma=0.01,                   # NEW: min loss reduction for split
            early_stopping_rounds=150,    # More patience
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ))
    ])
    
    # B. CatBoost — TUNED: lower lr, more iterations
    X_cat = X_all.copy()
    X_test_cat = X_test_all.copy()
    cat_features_indices = []
    for col in categorical_cols:
        X_cat[col] = X_cat[col].astype(str).fillna("Missing")
        X_test_cat[col] = X_test_cat[col].astype(str).fillna("Missing")
        cat_features_indices.append(col)
    for col in numeric_cols:
        X_cat[col] = X_cat[col].fillna(0)
        X_test_cat[col] = X_test_cat[col].fillna(0)
        
    model_cat = CatBoostRegressor(
        loss_function="MAE",
        iterations=6000,                  # More iterations (ES will cut)
        learning_rate=0.02,               # Lower LR
        depth=6,
        l2_leaf_reg=5,                    # Stronger regularization
        bagging_temperature=0.5,          # NEW: less randomness for stability
        random_strength=0.5,              # NEW: reduce random noise in splits
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        silent=True
    )
    
    # C. ExtraTrees — TUNED: more trees, deeper
    model_et = Pipeline([
        ("preprocessor", preprocessor),
        ("et", ExtraTreesRegressor(
            n_estimators=800,             # More trees
            max_depth=20,                 # Deeper
            min_samples_split=3,          # Finer splits
            min_samples_leaf=2,           # NEW: prevent very small leaves
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    # D. LightGBM — TUNED: refined params
    model_lgb = Pipeline([
        ("preprocessor", preprocessor),
        ("lgb", LGBMRegressor(
            n_estimators=8000,            # More trees (ES will cut)
            learning_rate=0.008,          # Lower LR
            max_depth=4,                  # Shallower
            num_leaves=20,                # Fewer leaves (was 31) = less overfitting
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.2,                # Stronger L1
            reg_lambda=2.0,               # Stronger L2
            min_child_samples=10,         # NEW: min data in leaves
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ))
    ])
    
    # 9. Generate OOFs
    oof_xgb, test_xgb = get_oof_preds("XGBoost", model_xgb, X_all, y_log, X_test_all, kf)
    oof_cat, test_cat = get_oof_preds("CatBoost", model_cat, X_cat, y_log, X_test_cat, kf, cat_features=cat_features_indices)
    oof_et, test_et = get_oof_preds("ExtraTrees", model_et, X_all, y_log, X_test_all, kf)
    oof_lgb, test_lgb = get_oof_preds("LightGBM", model_lgb, X_all, y_log, X_test_all, kf)
    
    # 10. Stacking — Try both Ridge and ElasticNet, keep best
    X_meta_train = pd.DataFrame({
        "XGB": oof_xgb, "Cat": oof_cat, "ET": oof_et, "LGB": oof_lgb
    })
    X_meta_test = pd.DataFrame({
        "XGB": test_xgb, "Cat": test_cat, "ET": test_et, "LGB": test_lgb
    })
    
    print(f"\n{'='*50}")
    print("Evaluating Meta-Models...")
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta_train, y_log)
    ridge_pred = np.expm1(ridge.predict(X_meta_train))
    ridge_mae = mean_absolute_error(np.expm1(y_log), ridge_pred)
    
    # ElasticNet (mix of L1 and L2)
    enet = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE)
    enet.fit(X_meta_train, y_log)
    enet_pred = np.expm1(enet.predict(X_meta_train))
    enet_mae = mean_absolute_error(np.expm1(y_log), enet_pred)
    
    print(f"Ridge MAE: {ridge_mae:.2f}, Coefs: {dict(zip(X_meta_train.columns, ridge.coef_.round(4)))}")
    print(f"ElasticNet MAE: {enet_mae:.2f}, Coefs: {dict(zip(X_meta_train.columns, enet.coef_.round(4)))}")
    
    # Pick the best meta-model
    if enet_mae < ridge_mae:
        meta_model = enet
        meta_name = "ElasticNet"
        mae = enet_mae
    else:
        meta_model = ridge
        meta_name = "Ridge"
        mae = ridge_mae
    
    print(f"\n>>> Best Meta-Model: {meta_name} (MAE={mae:.2f})")
    
    y_true = np.expm1(y_log)
    mape = mean_absolute_percentage_error(y_true, np.expm1(meta_model.predict(X_meta_train)))
    coefs = dict(zip(X_meta_train.columns, meta_model.coef_.round(4)))
    
    print(f"\n*** FINAL RESULTS ***")
    print(f"Tuned Stack OOF MAE: {mae:.4f}")
    print(f"Tuned Stack OOF MAPE: {mape*100:.2f}%")
    print(f"Meta: {meta_name}, Coefs: {coefs}")
    print(f"(RUN 011 reference: MAE=14020.51, Kaggle=12969.71)")
    
    # 11. Submission
    final_pred = np.expm1(meta_model.predict(X_meta_test))
    
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": final_pred
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to {SUBMISSION_PATH}")
    
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Tuned Stack MAE: {mae:.4f}\n")
        f.write(f"Tuned Stack MAPE: {mape:.4f}\n")
        f.write(f"Meta: {meta_name}, Coefs: {coefs}\n")
        f.write(f"Ridge MAE: {ridge_mae:.4f}, ElasticNet MAE: {enet_mae:.4f}\n")
        f.write(f"Improvements: Tuned HP (all 4 models), ElasticNet vs Ridge comparison\n")

if __name__ == "__main__":
    main()
