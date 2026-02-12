import pandas as pd
import numpy as np


def encode_ordinals(df):
    """
    Encode les variables qualité ordinales en valeurs numériques.
    Préserve l'information d'ordre (Ex > Gd > TA > Fa > Po) perdue par le OneHot.
    """
    df_out = df.copy()

    # Mapping standard pour les variables Qual/Cond (5 niveaux)
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC"
    ]
    for col in qual_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(qual_map).fillna(0).astype(int)

    # BsmtExposure (4 niveaux)
    bsmt_exp_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    if "BsmtExposure" in df_out.columns:
        df_out["BsmtExposure"] = df_out["BsmtExposure"].map(bsmt_exp_map).fillna(0).astype(int)

    # BsmtFinType (6 niveaux)
    bsmt_fin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(bsmt_fin_map).fillna(0).astype(int)

    # GarageFinish (3 niveaux)
    garage_fin_map = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    if "GarageFinish" in df_out.columns:
        df_out["GarageFinish"] = df_out["GarageFinish"].map(garage_fin_map).fillna(0).astype(int)

    # Fence
    fence_map = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    if "Fence" in df_out.columns:
        df_out["Fence"] = df_out["Fence"].map(fence_map).fillna(0).astype(int)

    # Functional (deductions)
    func_map = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}
    if "Functional" in df_out.columns:
        df_out["Functional"] = df_out["Functional"].map(func_map).fillna(8).astype(int)

    # PavedDrive
    paved_map = {"N": 0, "P": 1, "Y": 2}
    if "PavedDrive" in df_out.columns:
        df_out["PavedDrive"] = df_out["PavedDrive"].map(paved_map).fillna(0).astype(int)

    # CentralAir
    if "CentralAir" in df_out.columns:
        df_out["CentralAir"] = (df_out["CentralAir"] == "Y").astype(int)

    # LotShape
    lot_map = {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}
    if "LotShape" in df_out.columns:
        df_out["LotShape"] = df_out["LotShape"].map(lot_map).fillna(0).astype(int)

    # LandSlope
    slope_map = {"Sev": 0, "Mod": 1, "Gtl": 2}
    if "LandSlope" in df_out.columns:
        df_out["LandSlope"] = df_out["LandSlope"].map(slope_map).fillna(2).astype(int)

    return df_out


def add_features(df):
    """
    Ajoute des features ingéniées au DataFrame.
    Gère les NaNs pour les colonnes utilisées dans les calculs (remplacement par 0).
    
    Features créées:
    - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    - TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
    - Age = YrSold - YearBuilt
    - RemodAge = YrSold - YearRemodAdd
    - TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
    - HasPool = (PoolArea > 0)
    - HasGarage = (GarageArea > 0)
    - HasBasement = (TotalBsmtSF > 0)
    - HasFireplace = (Fireplaces > 0)
    """
    df_out = df.copy()
    
    # Remplacement des NaNs par 0 pour les surfaces et compteurs avant addition
    cols_fillna_0 = [
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
        "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
        "PoolArea", "GarageArea", "Fireplaces"
    ]
    
    # On remplit les NaNs par 0 uniquement pour le calcul, pour ne pas modifier les colonnes originales si le SimpleImputer doit passer après
    # Cependant, ici on crée de NOUVELLES colonnes, donc on peut utiliser des variables temporaires
    temp_df = df_out[cols_fillna_0].fillna(0)
    
    # 1. Surfaces Totales
    df_out["TotalSF"] = temp_df["TotalBsmtSF"] + temp_df["1stFlrSF"] + temp_df["2ndFlrSF"]
    
    # 2. Salles de bain totales
    df_out["TotalBath"] = (
        temp_df["FullBath"] + 
        0.5 * temp_df["HalfBath"] + 
        temp_df["BsmtFullBath"] + 
        0.5 * temp_df["BsmtHalfBath"]
    )
    
    # 3. Âge et Rénovation (YrSold et YearBuilt n'ont généralement pas de NaN, sinon propagation)
    df_out["Age"] = df_out["YrSold"] - df_out["YearBuilt"]
    df_out["RemodAge"] = df_out["YrSold"] - df_out["YearRemodAdd"]
    
    # Gestion des valeurs négatives possibles si incohérence des données (ex: vendu avant construction)
    df_out["Age"] = df_out["Age"].apply(lambda x: max(0, x))
    df_out["RemodAge"] = df_out["RemodAge"].apply(lambda x: max(0, x))

    # 4. Total Porch
    df_out["TotalPorchSF"] = (
        temp_df["OpenPorchSF"] + 
        temp_df["EnclosedPorch"] + 
        temp_df["3SsnPorch"] + 
        temp_df["ScreenPorch"]
    )
    
    # 5. Indicateurs binaires
    df_out["HasPool"] = (temp_df["PoolArea"] > 0).astype(int)
    df_out["HasGarage"] = (temp_df["GarageArea"] > 0).astype(int)
    df_out["HasBasement"] = (temp_df["TotalBsmtSF"] > 0).astype(int)
    df_out["HasFireplace"] = (temp_df["Fireplaces"] > 0).astype(int)
    
    # 6. Interactions Qualité × Surface (top features Kaggle)
    if "OverallQual" in df_out.columns:
        oq = df_out["OverallQual"].fillna(0)
        df_out["QualSF"] = oq * df_out["TotalSF"]
        if "GrLivArea" in df_out.columns:
            df_out["QualFinishSF"] = oq * df_out["GrLivArea"].fillna(0)
        if "OverallCond" in df_out.columns:
            df_out["OverallScore"] = oq * df_out["OverallCond"].fillna(0)
    
    # 7. Interactions Garage
    if "GarageCars" in df_out.columns and "GarageArea" in df_out.columns:
        df_out["GarageInteraction"] = df_out["GarageCars"].fillna(0) * temp_df["GarageArea"]
    
    # 8. Ratios (avec protection division par 0)
    if "BsmtFinSF1" in df_out.columns and "TotalBsmtSF" in df_out.columns:
        total_bsmt = df_out["TotalBsmtSF"].fillna(0).replace(0, 1)
        df_out["BsmtFinRatio"] = df_out["BsmtFinSF1"].fillna(0) / total_bsmt
        
    if "GrLivArea" in df_out.columns and "LotArea" in df_out.columns:
        lot = df_out["LotArea"].fillna(0).replace(0, 1)
        df_out["LivAreaRatio"] = df_out["GrLivArea"].fillna(0) / lot
    
    # 9. Surface extérieure totale
    if "WoodDeckSF" in df_out.columns:
        df_out["TotalOutdoorSF"] = df_out["WoodDeckSF"].fillna(0) + df_out["TotalPorchSF"]
    
    # 10. Scores composites (si ordinal encoding appliqué EN AMONT)
    # BsmtQual et GarageQual seront numériques si encode_ordinals() a été appelé avant
    if "BsmtQual" in df_out.columns and df_out["BsmtQual"].dtype in ["int64", "float64", "int32"]:
        df_out["BsmtScore"] = df_out["BsmtQual"] * df_out["TotalBsmtSF"].fillna(0)
    if "GarageQual" in df_out.columns and df_out["GarageQual"].dtype in ["int64", "float64", "int32"]:
        df_out["GarageScore"] = df_out["GarageQual"] * temp_df["GarageArea"]
    if "ExterQual" in df_out.columns and df_out["ExterQual"].dtype in ["int64", "float64", "int32"]:
        df_out["ExterScore"] = df_out["ExterQual"] * df_out["TotalSF"]
    
    return df_out

def add_poly_features(df):
    """
    Ajoute des features polynomiales (carrés) pour les surfaces.
    Doit être appliqué APRES add_features (car utilise TotalSF).
    """
    df_out = df.copy()
    
    # Liste des colonnes à élever au carré
    # On utilise des patterns ou une liste explicite
    area_cols = [
        "GrLivArea", 
        "TotalBsmtSF", 
        "1stFlrSF", 
        "2ndFlrSF", 
        "GarageArea", 
        "LotArea", 
        "MasVnrArea",
        "TotalSF" # Créée par add_features
    ]
    
    for col in area_cols:
        if col in df_out.columns:
            # On remplit les NaNs par 0 avant le carré pour éviter la propagation de NaN
            # On cast en float pour éviter l'overflow si entiers trop grands
            series_filled = df_out[col].fillna(0).astype(float)
            df_out[f"{col}_sq"] = series_filled ** 2
            
    # Interaction simple (si dispo)
    if "TotalSF" in df_out.columns and "OverallQual" in df_out.columns:
        df_out["TotalSF_x_OverallQual"] = df_out["TotalSF"].fillna(0) * df_out["OverallQual"].fillna(0)
        
    return df_out
