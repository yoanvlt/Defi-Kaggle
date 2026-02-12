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
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import add_features, add_poly_features

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_007_xgb_poly_area.csv"
RESULTS_PATH = "results_run007.txt"
RANDOM_STATE = 42
N_SPLITS = 5

def main():
    print(f"Run 007: XGB Poly Area started at {datetime.now()}")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df
    
    # 2. Pipeline Setup
    
    # A. Feature Engineering (Chained)
    # On utilise une fonction wrapper pour enchaîner les deux
    def separate_feature_engineering(df):
        df_1 = add_features(df)
        df_2 = add_poly_features(df_1)
        return df_2

    # B. Preprocessing
    # On doit définir les colonnes num/cat APRES le feature engineering.
    # Pour la construction du ColumnTransformer, on peut tricher en appliquant la transfo sur un sample
    sample_trans = separate_feature_engineering(X.head(5))
    numeric_cols = sample_trans.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = sample_trans.select_dtypes(include=["object"]).columns
    
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    new_cols = [c for c in sample_trans.columns if "_sq" in c or "_x_" in c]
    print(f"New Poly Features: {new_cols}")

    preprocessor = ColumnTransformer([
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
        ("fe", FunctionTransformer(separate_feature_engineering, validate=False)),
        ("preprocessor", preprocessor),
        ("model", xgb)
    ])
    
    final_model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # 3. CV Evaluation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    mae_scores = []
    mape_scores = []
    
    print("Starting Main CV...")
    # On ne peut pas utiliser cross_val_score facilement avec TransformedTargetRegressor si on veut le MAE sur l'échelle originale
    # mais cross_val_score le fait si le scoring est 'neg_mean_absolute_error' ET que l'estimator est TransformedTargetRegressor ?
    # Oui, TransformedTargetRegressor inverse la transformation avant de scorer.
    
    # Faisons une boucle manuelle pour être sûr et avoir le MAPE
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
    
    # 4. Final Training & Submission
    print("Retraining on full data...")
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
        f.write(f"Parameters: XGB Run 003 Best\n")
        f.write(f"New Features: {new_cols}\n")
        f.write(f"CV MAE: {mean_mae:.4f} (+/- {std_mae:.4f})\n")
        f.write(f"CV MAPE: {mean_mape:.4f}\n")

if __name__ == "__main__":
    main()
