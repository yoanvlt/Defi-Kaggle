import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# Import XGBoost or fallback
try:
    from xgboost import XGBRegressor
    MODEL_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    MODEL_TYPE = "HistGradientBoosting"

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_002_log_target_boosting.csv"
RESULTS_PATH = "results_run002.txt"
RANDOM_STATE = 42
N_SPLITS = 5

def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Les fichiers de données sont introuvables.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def main():
    print(f"Run 002: Log Target Boosting ({MODEL_TYPE}) started at {datetime.now()}")
    
    # 1. Load
    train_df, test_df = load_data()
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df

    # 2. Preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 3. Model
    if MODEL_TYPE == "XGBoost":
        base_model = XGBRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        print("XGBoost not found, using fallback.")
        base_model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    # 4. Target Transformation (Log1p)
    model_final = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # 5. Evaluate CV
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model_final, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    mae_scores = -scores

    print(f"Mean MAE: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}")
    
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Model: {MODEL_TYPE} + LogTarget\n")
        f.write(f"Mean MAE: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}\n")
        f.write(f"Full scores: {mae_scores}\n")

    # 6. Submission
    model_final.fit(X, y)
    predictions = model_final.predict(X_test)
    
    submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": predictions
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
