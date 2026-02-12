import pandas as pd
import numpy as np

def clean_data(df):
    """
    Applique un nettoyage méticuleux des données.
    Basé sur la documentation du dataset Ames Housing.
    """
    df_out = df.copy()
    
    # 1. Imputation "None" pour les variables catégorielles où NaN = "Pas d'équipement"
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna("None")
            
    # 2. Imputation 0 pour les variables numériques où NaN = "Pas d'équipement"
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
        "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0)
            
    # 3. Imputation Mode (Valeur la plus fréquente) pour les quelques manquants restants
    mode_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
    for col in mode_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(df_out[col].mode()[0])
            
    # 4. Functional : "Typ" par défaut
    if "Functional" in df_out.columns:
        df_out["Functional"] = df_out["Functional"].fillna("Typ")
        
    # 5. LotFrontage : Imputation par la médiane du voisinage
    # C'est une imputation "groupée", donc attention au data leakage si fait sur tout le set.
    # Ici on le fait ligne par ligne ou via transform
    if "LotFrontage" in df_out.columns and "Neighborhood" in df_out.columns:
        df_out["LotFrontage"] = df_out.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        # Si encore des NaNs (voisinage sans LotFrontage ?) -> Mediane globale
        df_out["LotFrontage"] = df_out["LotFrontage"].fillna(df_out["LotFrontage"].median())
        
    # 6. Correction de Types
    # MSSubClass est un code, pas une quantité
    if "MSSubClass" in df_out.columns:
        df_out["MSSubClass"] = df_out["MSSubClass"].astype(str)
        
    # YrSold et MoSold sont des catégories temporelles (saisonnalité), pas des quantités continues linéaires
    # MAIS features.py a besoin de YrSold en numérique pour calculer Age = YrSold - YearBuilt
    # Donc on ne les convertit pas en string ici. On le fera dans le pipeline de preprocessing via OneHotEncoder
    # en spécifiant les colonnes manuellement si besoin, ou on laisse le modèle gérer.
    # Pour l'instant, on commente cette conversion qui casse features.py
    
    # if "YrSold" in df_out.columns:
    #     df_out["YrSold"] = df_out["YrSold"].astype(str)
    # if "MoSold" in df_out.columns:
    #     df_out["MoSold"] = df_out["MoSold"].astype(str)
        
    return df_out

def remove_outliers(df):
    """
    Supprime les outliers identifiés dans le training set.
    """
    df_out = df.copy()
    
    # GrLivArea > 4000 et SalePrice < 300000 (recommandé par l'auteur du dataset)
    outliers_idx = df_out[(df_out["GrLivArea"] > 4000) & (df_out["SalePrice"] < 300000)].index
    df_out = df_out.drop(outliers_idx)
    print(f"Removed {len(outliers_idx)} outliers.")
    
    return df_out
