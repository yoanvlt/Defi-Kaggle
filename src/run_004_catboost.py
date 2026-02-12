import pandas as pd
import numpy as np
import os
import random
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_004_catboost.csv"
RESULTS_PATH = "results_run004.txt"
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_CV = 1 # Single optimized run

def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Data files not found.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def prepare_data(train_df, test_df):
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df.copy()
    
    cat_features = list(X.select_dtypes(include=["object"]).columns)
    
    # Remplacement simple des NaNs catégoriels pour CatBoost
    for col in cat_features:
        X[col] = X[col].astype(str).fillna("Missing")
        X_test[col] = X_test[col].astype(str).fillna("Missing")
        
    y_log = np.log1p(y)
    
    return X, y_log, X_test, cat_features

def train_evaluate_cv(params, X, y_log, cat_features):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mae_scores = []
    mape_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_log, y_val_log = y_log.iloc[train_index], y_log.iloc[val_index]
        
        train_pool = Pool(X_train, y_train_log, cat_features=cat_features)
        val_pool = Pool(X_val, y_val_log, cat_features=cat_features)
        
        # 'allow_writing_files=False' désactive la création des dossiers catboost_info
        model = CatBoostRegressor(**params, silent=True, allow_writing_files=False)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200, verbose=False)
        
        preds_log = model.predict(X_val)
        preds = np.expm1(preds_log)
        y_val_true = np.expm1(y_val_log)
        
        mae_scores.append(mean_absolute_error(y_val_true, preds))
        mape_scores.append(mean_absolute_percentage_error(y_val_true, preds))
        
    return np.mean(mae_scores), np.mean(mape_scores), np.std(mae_scores)

def main():
    print(f"Run 004: CatBoost started at {datetime.now()}")
    
    # 1. Load & Prepare
    train_df, test_df = load_data()
    X, y_log, X_test, cat_features = prepare_data(train_df, test_df)
    
    print(f"Categorical features: {len(cat_features)}")
    
    # 2. Manual Tuning (Single Run with Best Guess)
    param_grid = {
        'depth': [6],
        'learning_rate': [0.03],
        'l2_leaf_reg': [3],
        'random_strength': [1],
        'iterations': [5000] # Large number, relying on early stopping
    }
    
    best_mae = float('inf')
    best_mape = float('inf')
    best_mae_std = 0
    best_params = {}
    
    print(f"Starting Single Run (Manual Tuning)...")
    random.seed(RANDOM_STATE)
    
    # Loop of 1
    for i in range(N_ITER_CV):
        # Pick 0th element since lists have len 1
        params = {
            'loss_function': 'MAE',
            'random_seed': RANDOM_STATE,
            'iterations': param_grid['iterations'][0],
            'depth': param_grid['depth'][0],
            'learning_rate': param_grid['learning_rate'][0],
            'l2_leaf_reg': param_grid['l2_leaf_reg'][0],
            'random_strength': param_grid['random_strength'][0]
        }
        
        mae, mape, std = train_evaluate_cv(params, X, y_log, cat_features)
        print(f"Run 1: MAE={mae:.2f}, MAPE={mape*100:.2f}%, Params={params}")
        
        if mae < best_mae:
            best_mae = mae
            best_mape = mape
            best_mae_std = std
            best_params = params
            
    print(f"\nBest MAE: {best_mae:.4f} +/- {best_mae_std:.4f}")
    print(f"Best MAPE: {best_mape*100:.2f}%")
    print(f"Best Params: {best_params}")
    
    # Write results
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Mean MAE: {best_mae:.4f} +/- {best_mae_std:.4f}\n")
        f.write(f"Mean MAPE: {best_mape*100:.2f}%\n")
    
    # 3. Final Submission
    print("\nGenerating submission...")
    full_pool = Pool(X, y_log, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)
    
    final_model = CatBoostRegressor(**best_params, silent=True, allow_writing_files=False)
    final_model.fit(full_pool)
    
    preds_log = final_model.predict(test_pool)
    preds = np.expm1(preds_log)
    
    submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": preds
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
