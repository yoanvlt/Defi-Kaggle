import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Importer add_features depuis src/features.py
# On suppose que le script est lancé depuis la racine du projet
sys.path.append("src")
try:
    from features import add_features
except ImportError:
    # Fallback si lancé depuis src/
    sys.path.append(".")
    from features import add_features

from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor

# --- CONFIGURATION ---
TRAIN_PATH = "Data/train.csv"
TEST_PATH = "Data/test.csv"
SUBMISSION_PATH = "submissions/submission_003_fe_tuned_xgb.csv"
RESULTS_PATH = "results_run003.txt"
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_SEARCH = 20

def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Data files not found.")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def main():
    print(f"Run 003: Feature Engineering & Tuning started at {datetime.now()}")
    
    # 1. Load Data
    train_df, test_df = load_data()
    X = train_df.drop(columns=["SalePrice"])
    y = train_df["SalePrice"]
    X_test = test_df
    
    # 2. Setup Pipeline
    # Identifier les colonnes numériques/catégorielles initiales
    numeric_features = list(X.select_dtypes(include=["int64", "float64"]).columns)
    categorical_features = list(X.select_dtypes(include=["object"]).columns)
    
    # Nouvelles colonnes créées par add_features
    new_cols = [
        "TotalSF", "TotalBath", "Age", "RemodAge", "TotalPorchSF", 
        "HasPool", "HasGarage", "HasBasement", "HasFireplace"
    ]
    
    # On ajoute les nouvelles colonnes à la liste des numériques
    # (car add_features renvoie tout le df enrichi, et ces nouvelles features sont numériques)
    full_numeric_features = numeric_features + new_cols
    
    # Preprocessing
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, full_numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Modèle
    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Pipeline Base
    pipeline = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_features, validate=False)),
        ("preprocessor", preprocessor),
        ("model", xgb)
    ])
    
    # Log Transformation
    final_model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # 3. Randomized Search
    print("Starting interaction-restricted RandomizedSearchCV...")
    # Paramètres pour TransformedTargetRegressor -> regressor -> model
    param_dist = {
        "regressor__model__n_estimators": [1500, 2500, 4000, 6000],
        "regressor__model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "regressor__model__max_depth": [3, 4, 5],
        "regressor__model__min_child_weight": [1, 3, 5],
        "regressor__model__subsample": [0.6, 0.75, 0.85, 1.0],
        "regressor__model__colsample_bytree": [0.6, 0.75, 0.85, 1.0],
        "regressor__model__reg_alpha": [0.0, 0.1, 0.5],
        "regressor__model__reg_lambda": [0.5, 1.0, 2.0]
    }
    
    search = RandomizedSearchCV(
        estimator=final_model,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="neg_mean_absolute_error",
        cv=5, # CV interne pour le tuning
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X, y)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    print(f"Best Params: {best_params}")
    print(f"Best CV MAE (internal): {-search.best_score_:.4f}")
    
    # 4. Final Evaluation (Outer CV Loop - Clean Evaluation)
    # On réévalue le meilleur modèle proprement sur 5 folds
    print("\n--- Final CV Evaluation (MAE + MAPE) ---")
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error']
    
    scores = cross_validate(best_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    mae = -scores['test_neg_mean_absolute_error']
    mape = -scores['test_neg_mean_absolute_percentage_error']
    
    print(f"MAE: {mae.mean():.4f} +/- {mae.std():.4f}")
    print(f"MAPE: {mape.mean()*100:.2f}% +/- {mape.std()*100:.2f}%")
    
    # Write results
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Mean MAE: {mae.mean():.4f} +/- {mae.std():.4f}\n")
        f.write(f"Mean MAPE: {mape.mean()*100:.2f}% +/- {mape.std()*100:.2f}%\n")
    
    # 5. Submission
    print("\nGenerating submission...")
    best_model.fit(X, y)
    predictions = best_model.predict(X_test)
    
    submission = pd.DataFrame({
        "Id": X_test["Id"],
        "SalePrice": predictions
    })
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
