import pandas as pd
import numpy as np

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
