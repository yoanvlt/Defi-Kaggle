import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from scipy.stats import skew
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, Lasso
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
SUBMISSION_PATH = "submissions/submission_016_feature_boost.csv"
RESULTS_PATH = "results_run016.txt"
RANDOM_STATE = 42
N_SPLITS = 5

# Identique RUN 013
DROP_FEATURES = ["Utilities", "Street", "PoolQC", "PoolArea", "MiscFeature", "MiscVal", "Id"]
TARGET_ENCODE_COLS = ["Neighborhood", "Condition1", "Exterior1st", "Exterior2nd"]
SKEW_THRESHOLD = 0.75


# ==========================================================
# NOUVEAU : Features métier supplémentaires
# ==========================================================

def add_advanced_features(df):
    """
    Ajoute des features métier que les modèles ne découvrent pas seuls.
    Doit être appelé APRES encode_ordinals() et add_features().
    """
    df_out = df.copy()

    # 1. HasBeenRemodeled : la maison a-t-elle été rénovée ?
    if "YearRemodAdd" in df_out.columns and "YearBuilt" in df_out.columns:
        df_out["HasBeenRemodeled"] = (df_out["YearRemodAdd"] != df_out["YearBuilt"]).astype(int)

    # 2. BathPerSF : densité de salles de bain (proxy luxe)
    if "TotalBath" in df_out.columns and "TotalSF" in df_out.columns:
        total_sf = df_out["TotalSF"].replace(0, 1)  # Protection division par 0
        df_out["BathPerSF"] = df_out["TotalBath"] / total_sf

    # 3. KitchenValue : qualité x nombre de cuisines
    if "KitchenQual" in df_out.columns and "KitchenAbvGr" in df_out.columns:
        kq = df_out["KitchenQual"]
        if kq.dtype in ["int64", "int32", "float64"]:
            df_out["KitchenValue"] = kq * df_out["KitchenAbvGr"].fillna(1)

    # 4. TotalQualScore : somme de tous les scores qualité ordinaux
    qual_cols = ["OverallQual", "ExterQual", "KitchenQual", "BsmtQual"]
    existing_qual = [c for c in qual_cols if c in df_out.columns and df_out[c].dtype in ["int64", "int32", "float64"]]
    if existing_qual:
        df_out["TotalQualScore"] = df_out[existing_qual].fillna(0).sum(axis=1)

    # 5. SeasonSold : printemps (mars-juin) = haute saison immobilière
    if "MoSold" in df_out.columns:
        df_out["SeasonSold"] = df_out["MoSold"].apply(lambda m: 1 if m in [3, 4, 5, 6] else 0)

    # 6. Has2ndFloor : présence d'un étage
    if "2ndFlrSF" in df_out.columns:
        df_out["Has2ndFloor"] = (df_out["2ndFlrSF"].fillna(0) > 0).astype(int)

    # 7. GarageValue : capacité × finition du garage
    if "GarageCars" in df_out.columns and "GarageFinish" in df_out.columns:
        gc = df_out["GarageCars"].fillna(0)
        gf = df_out["GarageFinish"]
        if gf.dtype in ["int64", "int32", "float64"]:
            df_out["GarageValue"] = gc * gf

    # 8. AgeBin : tranches d'âge (neuf/récent/ancien/très ancien)
    if "Age" in df_out.columns:
        df_out["AgeBin"] = pd.cut(
            df_out["Age"],
            bins=[-1, 5, 15, 30, 50, 200],
            labels=[4, 3, 2, 1, 0]  # 4=neuf, 0=très ancien
        ).astype(float).fillna(0)

    # 9. BONUS: FinishedBsmtRatio — part du sous-sol aménagé
    if "BsmtFinSF1" in df_out.columns and "TotalBsmtSF" in df_out.columns:
        total_bsmt = df_out["TotalBsmtSF"].fillna(0).replace(0, 1)
        bsmt_fin = df_out["BsmtFinSF1"].fillna(0) + df_out.get("BsmtFinSF2", pd.Series(0, index=df_out.index)).fillna(0)
        df_out["FinishedBsmtRatio"] = bsmt_fin / total_bsmt

    # 10. BONUS: OverallGrade — OverallQual × OverallCond (qualité globale)
    if "OverallQual" in df_out.columns and "OverallCond" in df_out.columns:
        df_out["OverallGrade"] = df_out["OverallQual"] * df_out["OverallCond"]

    n_new = sum(1 for c in ["HasBeenRemodeled", "BathPerSF", "KitchenValue",
                             "TotalQualScore", "SeasonSold", "Has2ndFloor",
                             "GarageValue", "AgeBin", "FinishedBsmtRatio", "OverallGrade"]
                if c in df_out.columns)
    print(f"  Advanced features added: {n_new}")
    return df_out


def fix_skewness(df, skewed_cols=None, threshold=SKEW_THRESHOLD):
    numeric_cols = df.select_dtypes(include=["int64", "int32", "float64"]).columns
    if skewed_cols is None:
        skewed_feats = df[numeric_cols].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats.abs() > threshold]
        skewed_cols = [col for col in skewed_feats.index if (df[col].dropna() >= 0).all()]
        print(f"  Skewness correction: {len(skewed_cols)} features")
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df, skewed_cols


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
    is_linear = "Linear" in model_name

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
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            xgb_model.fit(X_train_p, y_train_log,
                          eval_set=[(X_val_p, y_val_log)], verbose=False)
            val_pred_log = xgb_model.predict(X_val_p)
            test_pred_log = xgb_model.predict(X_test_p)
        elif is_lgb:
            preprocessor = model.named_steps["preprocessor"]
            lgb_model = model.named_steps["lgb"]
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            lgb_model.fit(X_train_p, y_train_log,
                          eval_set=[(X_val_p, y_val_log)],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
            val_pred_log = lgb_model.predict(X_val_p)
            test_pred_log = lgb_model.predict(X_test_p)
        elif is_linear:
            preprocessor = model.named_steps["preprocessor"]
            linear_model = model.named_steps["linear"]
            X_train_p = preprocessor.fit_transform(X_train)
            X_val_p = preprocessor.transform(X_val)
            X_test_p = preprocessor.transform(X_test)
            linear_model.fit(X_train_p, y_train_log)
            val_pred_log = linear_model.predict(X_val_p)
            test_pred_log = linear_model.predict(X_test_p)
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


def build_linear_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])


def build_tree_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])


def main():
    print(f"Run 016: Feature Boost Stack started at {datetime.now()}")

    # 1. Load
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df["Id"].copy()

    # 2. Outlier removal — IDENTIQUE RUN 013 (pas l'agressif du RUN 015)
    print("Removing outliers (standard)...")
    original_len = len(train_df)
    train_df = remove_outliers(train_df)
    print(f"Train: {original_len} -> {len(train_df)}")

    # 3. Target
    y = train_df["SalePrice"]
    y_log = np.log1p(y)
    X_train_raw = train_df.drop(columns=["SalePrice"])
    X_test_raw = test_df.copy()

    # 4. Clean + Ordinals + FE + NOUVELLES FEATURES
    def preprocess_global(df):
        return add_advanced_features(add_features(encode_ordinals(clean_data(df))))

    print("\nPreprocessing (with advanced features)...")
    X_all = preprocess_global(X_train_raw)
    X_test_all = preprocess_global(X_test_raw)

    # 5. Target Encoding OOF (identique RUN 013)
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

    # 7. Skewness — train-aligned
    print("\nFixing skewness...")
    X_all, skewed_cols = fix_skewness(X_all)
    X_test_all, _ = fix_skewness(X_test_all, skewed_cols=skewed_cols)

    # 8. Types
    numeric_cols = X_all.select_dtypes(include=["int64", "int32", "float64"]).columns
    categorical_cols = X_all.select_dtypes(include=["object"]).columns
    print(f"\nFeatures: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    tree_preprocessor = build_tree_preprocessor(numeric_cols, categorical_cols)
    linear_preprocessor = build_linear_preprocessor(numeric_cols, categorical_cols)

    # --- A. XGBoost ---
    model_xgb = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("xgb", XGBRegressor(
            n_estimators=8000, learning_rate=0.008, max_depth=4,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=2.0,
            min_child_weight=3, gamma=0.01,
            early_stopping_rounds=150,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    # --- B. CatBoost ---
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
        loss_function="MAE", iterations=6000, learning_rate=0.02,
        depth=6, l2_leaf_reg=5, bagging_temperature=0.5,
        random_strength=0.5, random_seed=RANDOM_STATE,
        allow_writing_files=False, silent=True
    )

    # --- C. ExtraTrees ---
    model_et = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("et", ExtraTreesRegressor(
            n_estimators=800, max_depth=20,
            min_samples_split=3, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    # --- D. LightGBM ---
    model_lgb = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("lgb", LGBMRegressor(
            n_estimators=8000, learning_rate=0.008, max_depth=4,
            num_leaves=20, subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.2, reg_lambda=2.0, min_child_samples=10,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ))
    ])

    # --- E. Ridge ---
    model_ridge = Pipeline([
        ("preprocessor", linear_preprocessor),
        ("linear", Ridge(alpha=10.0, random_state=RANDOM_STATE))
    ])

    # --- F. Lasso ---
    model_lasso = Pipeline([
        ("preprocessor", linear_preprocessor),
        ("linear", Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE))
    ])

    # 9. OOFs
    oof_xgb, test_xgb = get_oof_preds("XGBoost", model_xgb, X_all, y_log, X_test_all, kf)
    oof_cat, test_cat = get_oof_preds("CatBoost", model_cat, X_cat, y_log, X_test_cat, kf, cat_features=cat_features_indices)
    oof_et, test_et = get_oof_preds("ExtraTrees", model_et, X_all, y_log, X_test_all, kf)
    oof_lgb, test_lgb = get_oof_preds("LightGBM", model_lgb, X_all, y_log, X_test_all, kf)
    oof_ridge, test_ridge = get_oof_preds("Linear_Ridge", model_ridge, X_all, y_log, X_test_all, kf)
    oof_lasso, test_lasso = get_oof_preds("Linear_Lasso", model_lasso, X_all, y_log, X_test_all, kf)

    # 10. Meta-stacking
    X_meta_train = pd.DataFrame({
        "XGB": oof_xgb, "Cat": oof_cat, "ET": oof_et,
        "LGB": oof_lgb, "Ridge": oof_ridge, "Lasso": oof_lasso,
    })
    X_meta_test = pd.DataFrame({
        "XGB": test_xgb, "Cat": test_cat, "ET": test_et,
        "LGB": test_lgb, "Ridge": test_ridge, "Lasso": test_lasso,
    })

    print(f"\n{'='*50}")
    print("Evaluating Meta-Models (Feature Boost edition)...")

    # Ridge meta
    ridge_meta = Ridge(alpha=1.0)
    ridge_meta.fit(X_meta_train, y_log)
    ridge_pred = np.expm1(ridge_meta.predict(X_meta_train))
    ridge_mae = mean_absolute_error(np.expm1(y_log), ridge_pred)

    # ElasticNet meta
    enet_meta = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE)
    enet_meta.fit(X_meta_train, y_log)
    enet_pred = np.expm1(enet_meta.predict(X_meta_train))
    enet_mae = mean_absolute_error(np.expm1(y_log), enet_pred)

    print(f"Ridge    meta MAE: {ridge_mae:.2f} | Coefs: {dict(zip(X_meta_train.columns, ridge_meta.coef_.round(4)))}")
    print(f"ElasticNet meta MAE: {enet_mae:.2f} | Coefs: {dict(zip(X_meta_train.columns, enet_meta.coef_.round(4)))}")

    if enet_mae < ridge_mae:
        meta_model = enet_meta
        meta_name = "ElasticNet"
        mae = enet_mae
    else:
        meta_model = ridge_meta
        meta_name = "Ridge"
        mae = ridge_mae

    print(f"\n>>> Best Meta-Model: {meta_name} (MAE={mae:.2f})")

    y_true = np.expm1(y_log)
    mape = mean_absolute_percentage_error(y_true, np.expm1(meta_model.predict(X_meta_train)))
    coefs = dict(zip(X_meta_train.columns, meta_model.coef_.round(4)))

    print(f"\n*** FINAL RESULTS RUN 016 ***")
    print(f"Stack OOF MAE  : {mae:.4f}")
    print(f"Stack OOF MAPE : {mape*100:.2f}%")
    print(f"Meta: {meta_name}, Coefs: {coefs}")
    print(f"(RUN 013 reference: MAE=13263.91, Kaggle=12608.33)")

    # 11. Submission
    final_pred = np.expm1(meta_model.predict(X_meta_test))
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_pred})
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to {SUBMISSION_PATH}")

    # 12. Log
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"RUN 016 - Feature Boost Stack\n")
        f.write(f"New features: HasBeenRemodeled, BathPerSF, KitchenValue, TotalQualScore, SeasonSold, Has2ndFloor, GarageValue, AgeBin, FinishedBsmtRatio, OverallGrade\n")
        f.write(f"Stack MAE : {mae:.4f}\n")
        f.write(f"Stack MAPE: {mape:.4f}\n")
        f.write(f"Meta      : {meta_name}, Coefs: {coefs}\n")
        f.write(f"Ridge meta MAE: {ridge_mae:.4f}, ElasticNet meta MAE: {enet_mae:.4f}\n")
        f.write(f"RUN 013 reference: MAE=13263.91, Kaggle=12608.33\n")


if __name__ == "__main__":
    main()
