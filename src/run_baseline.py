import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_001_baseline.csv"
RANDOM_STATE = 42
N_SPLITS = 5

def load_data():
    """Charge les données train et test."""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Les fichiers de données sont introuvables. Vérifiez {TRAIN_PATH} et {TEST_PATH}.")
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Data loaded: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df

def prepare_data(train_df, test_df):
    """Sépare les features et la cible, et identifie les types de colonnes."""
    # Séparation X/y
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df # Pas de cible dans le test set

    # Identification des colonnes
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    print(f"Numerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    return X, y, X_test, numeric_features, categorical_features

def build_pipeline(numeric_features, categorical_features):
    """Construit la pipeline de preprocessing et le modèle."""
    # Preprocessing
    numeric_transformer = SimpleImputer(strategy="median")
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # sparse=False pour HistGradientBoosting
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Modèle Baseline
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    # Pipeline complète
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline

def evaluate_model(pipeline, X, y):
    """Évalue le modèle avec une validation croisée (MAE)."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # neg_mean_absolute_error car scikit-learn maximise le score
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    mae_scores = -scores # On repasse en positif
    
    print("\n--- Cross-Validation Results ---")
    for i, score in enumerate(mae_scores):
        print(f"Fold {i+1}: MAE = {score:.4f}")
        
    print(f"\nMean MAE: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}")
    
    with open("results.txt", "w") as f:
        f.write(f"Mean MAE: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}\n")
        for i, score in enumerate(mae_scores):
            f.write(f"Fold {i+1}: {score:.4f}\n")
            
    return mae_scores.mean(), mae_scores.std()

def generate_submission(pipeline, X, y, X_test):
    """Entraîne sur tout le train set et génère la soumission."""
    print("\n--- Training on full dataset and generating submission ---")
    pipeline.fit(X, y)
    predictions = pipeline.predict(X_test)
    
    submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": predictions
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

def main():
    print(f"Run 001: Baseline started at {datetime.now()}")
    
    # 1. Load Data
    train_df, test_df = load_data()
    
    # 2. Prepare Data
    X, y, X_test, num_feats, cat_feats = prepare_data(train_df, test_df)
    
    # 3. Build Pipeline
    pipeline = build_pipeline(num_feats, cat_feats)
    
    # 4. Evaluate
    evaluate_model(pipeline, X, y)
    
    # 5. Submit
    generate_submission(pipeline, X, y, X_test)

if __name__ == "__main__":
    main()
