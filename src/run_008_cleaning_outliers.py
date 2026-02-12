import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_features
from src.cleaning import clean_data, remove_outliers

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_008_cleaning_outliers.csv"
RESULTS_PATH = "results_run008.txt"
RANDOM_STATE = 42
N_SPLITS = 5

def main():
    print(f"Run 008: Advanced Cleaning & Outliers started at {datetime.now()}")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. Outlier Removal (TRAIN ONLY)
    print("Removing outliers from Train set...")
    original_len = len(train_df)
    train_df = remove_outliers(train_df)
    print(f"Train set size: {original_len} -> {len(train_df)}")
    
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df
    
    # 3. Pipeline Setup
    
    # A. Feature Engineering Chain
    def engineering_chain(df):
        # 1. Clean Data (Imputation fine, types)
        df_clean = clean_data(df)
        # 2. Add New Features (TotalSF, etc.)
        df_feat = add_features(df_clean)
        return df_feat

    # B. Preprocessing
    # Détection des colonnes sur un sample transformé
    sample_trans = engineering_chain(X.head(5))
    numeric_cols = sample_trans.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = sample_trans.select_dtypes(include=["object"]).columns
    
    print(f"Features after cleaning: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    preprocessor = ColumnTransformer([
        # Pour les numériques, on garde SimpleImputer(median) au cas où Cleaning en a laissé passer
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])
    
    # C. Model (XGB from RUN 003 best params)
    xgb = XGBRegressor(
        n_estimators=4000, 
        learning_rate=0.01, 
        max_depth=5, 
        subsample=0.75, 
        colsample_bytree=0.75,
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ("fe", FunctionTransformer(engineering_chain, validate=False)),
        ("preprocessor", preprocessor),
        ("model", xgb)
    ])
    
    final_model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # 4. CV Evaluation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    mae_scores = []
    mape_scores = []
    
    print("Starting Main CV...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        
        final_model.fit(X_t, y_t)
        preds = final_model.predict(X_v)
        
        mae = mean_absolute_error(y_v, preds)
        mape = mean_absolute_percentage_error(y_v, preds)
        
        mae_scores.append(mae)
        mape_scores.append(mape)
        print(f"  Fold {fold+1}: MAE={mae:.0f}, MAPE={mape:.4f}")
        
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    mean_mape = np.mean(mape_scores)
    
    print(f"CV Results: MAE={mean_mae:.2f} (+/- {std_mae:.2f}), MAPE={mean_mape*100:.2f}%")
    
    # 5. Final Training & Submission
    print("Retraining on full cleaned data...")
    final_model.fit(X, y)
    
    submission_preds = final_model.predict(X_test)
    
    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": submission_preds
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    
    # Save Results
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Description: Improved Cleaning + Outlier Removal + XGB\n")
        f.write(f"CV MAE: {mean_mae:.4f} (+/- {std_mae:.4f})\n")
        f.write(f"CV MAPE: {mean_mape:.4f}\n")

if __name__ == "__main__":
    main()
