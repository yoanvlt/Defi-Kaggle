"""
advanced_features.py — Features métier supplémentaires (RUN 016+).

Ces features exploitent la connaissance du domaine immobilier
pour injecter du signal que les modèles ne découvrent pas seuls.
Doit être appelé APRES encode_ordinals() et add_features().
"""

import pandas as pd
import numpy as np


def add_spatial_clusters(df):
    """
    Crée une feature 'Neighborhood_Clustered' qui regroupe les quartiers
    selon leur niveau de prix médian (Faible, Moyen-Faible, Moyen-Haut, Haut).
    """
    df_out = df.copy()
    
    if "Neighborhood" in df_out.columns:
        # Clusters définis selon l'observation des prix médians
        cluster_map = {
            "MeadowV": 0, "IDOTRR": 0, "BrDale": 0, "OldTown": 0, "Edwards": 0, "BrkSide": 0, "Sawyer": 0, "Blueste": 0, "SWISU": 0, # < 140k
            "NAmes": 1, "NPkVill": 1, "Mitchel": 1, "SawyerW": 1, "Gilbert": 1, "NWAmes": 1, "Blmngtn": 1, "CollgCr": 1, # 140k - 198k
            "ClearCr": 2, "Crawfor": 2, "Veenker": 2, "Somerst": 2, "Timber": 2, # 200k - 230k
            "StoneBr": 3, "NoRidge": 3, "NridgHt": 3 # > 270k
        }
        df_out["Neighborhood_Clustered"] = df_out["Neighborhood"].map(cluster_map).fillna(1).astype(int)
        print("  Spatial feature added: Neighborhood_Clustered")
        
    return df_out


def add_advanced_features(df):
    """
    Ajoute 10 features métier au DataFrame.
    
    Features créées :
        - HasBeenRemodeled : rénovation (YearRemodAdd != YearBuilt)
        - BathPerSF : densité salles de bain / surface
        - KitchenValue : qualité × nombre de cuisines
        - TotalQualScore : somme des scores qualité ordinaux
        - SeasonSold : haute saison immobilière (mars-juin)
        - Has2ndFloor : présence d'un étage
        - GarageValue : capacité × finition du garage
        - AgeBin : tranches d'âge (neuf/récent/ancien/très ancien)
        - FinishedBsmtRatio : part du sous-sol aménagé
        - OverallGrade : OverallQual × OverallCond
    """
    df_out = df.copy()

    # 1. HasBeenRemodeled
    if "YearRemodAdd" in df_out.columns and "YearBuilt" in df_out.columns:
        df_out["HasBeenRemodeled"] = (df_out["YearRemodAdd"] != df_out["YearBuilt"]).astype(int)

    # 2. BathPerSF
    if "TotalBath" in df_out.columns and "TotalSF" in df_out.columns:
        total_sf = df_out["TotalSF"].replace(0, 1)
        df_out["BathPerSF"] = df_out["TotalBath"] / total_sf

    # 3. KitchenValue
    if "KitchenQual" in df_out.columns and "KitchenAbvGr" in df_out.columns:
        kq = df_out["KitchenQual"]
        if kq.dtype in ["int64", "int32", "float64"]:
            df_out["KitchenValue"] = kq * df_out["KitchenAbvGr"].fillna(1)

    # 4. TotalQualScore
    qual_cols = ["OverallQual", "ExterQual", "KitchenQual", "BsmtQual"]
    existing_qual = [c for c in qual_cols if c in df_out.columns and df_out[c].dtype in ["int64", "int32", "float64"]]
    if existing_qual:
        df_out["TotalQualScore"] = df_out[existing_qual].fillna(0).sum(axis=1)

    # 5. SeasonSold
    if "MoSold" in df_out.columns:
        df_out["SeasonSold"] = df_out["MoSold"].apply(lambda m: 1 if m in [3, 4, 5, 6] else 0)

    # 6. Has2ndFloor
    if "2ndFlrSF" in df_out.columns:
        df_out["Has2ndFloor"] = (df_out["2ndFlrSF"].fillna(0) > 0).astype(int)

    # 7. GarageValue
    if "GarageCars" in df_out.columns and "GarageFinish" in df_out.columns:
        gc = df_out["GarageCars"].fillna(0)
        gf = df_out["GarageFinish"]
        if gf.dtype in ["int64", "int32", "float64"]:
            df_out["GarageValue"] = gc * gf

    # 8. AgeBin
    if "Age" in df_out.columns:
        df_out["AgeBin"] = pd.cut(
            df_out["Age"],
            bins=[-1, 5, 15, 30, 50, 200],
            labels=[4, 3, 2, 1, 0]
        ).astype(float).fillna(0)

    # 9. FinishedBsmtRatio
    if "BsmtFinSF1" in df_out.columns and "TotalBsmtSF" in df_out.columns:
        total_bsmt = df_out["TotalBsmtSF"].fillna(0).replace(0, 1)
        bsmt_fin = df_out["BsmtFinSF1"].fillna(0) + df_out.get("BsmtFinSF2", pd.Series(0, index=df_out.index)).fillna(0)
        df_out["FinishedBsmtRatio"] = bsmt_fin / total_bsmt

    # 10. OverallGrade
    if "OverallQual" in df_out.columns and "OverallCond" in df_out.columns:
        df_out["OverallGrade"] = df_out["OverallQual"] * df_out["OverallCond"]

    n_new = sum(1 for c in ["HasBeenRemodeled", "BathPerSF", "KitchenValue",
                             "TotalQualScore", "SeasonSold", "Has2ndFloor",
                             "GarageValue", "AgeBin", "FinishedBsmtRatio", "OverallGrade"]
                if c in df_out.columns)
    print(f"  Advanced features added: {n_new}")
    return df_out
