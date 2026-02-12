import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
SUBMISSION_XGB_PATH = "submissions/submission_003_fe_tuned_xgb.csv"
SUBMISSION_CAT_PATH = "submissions/submission_004_catboost.csv"
OUTPUT_DIR = "submissions"

def main():
    print(f"Run 005: Blending started at {datetime.now()}")
    
    # 1. Load Submissions
    if not os.path.exists(SUBMISSION_XGB_PATH) or not os.path.exists(SUBMISSION_CAT_PATH):
        raise FileNotFoundError("Input submission files not found.")
        
    df_xgb = pd.read_csv(SUBMISSION_XGB_PATH)
    df_cat = pd.read_csv(SUBMISSION_CAT_PATH)
    
    # 2. Verify Integrity
    if df_xgb.shape != df_cat.shape:
        raise ValueError("Submissions have different shapes.")
    
    # Ensure sorted by Id
    df_xgb = df_xgb.sort_values("Id").reset_index(drop=True)
    df_cat = df_cat.sort_values("Id").reset_index(drop=True)
    
    if not df_xgb["Id"].equals(df_cat["Id"]):
        raise ValueError("Submission Ids do not match.")
        
    print(f"Submissions loaded and verified. Rows: {len(df_xgb)}")
    print(f"XGB Range: {df_xgb['SalePrice'].min():.2f} - {df_xgb['SalePrice'].max():.2f}")
    print(f"Cat Range: {df_cat['SalePrice'].min():.2f} - {df_cat['SalePrice'].max():.2f}")
    
    # 3. Create Blends
    # Weights for XGB (since it performed better on Leaderboard & CV than untuned CatBoost, presumably)
    # Actually RUN 003 LB was 14125 vs CV 14806. CatBoost RUN 004 CV was 15264.
    # So XGB is stronger. We test high weights for XGB.
    weights_xgb = [0.9, 0.8, 0.7, 0.6]
    
    for w in weights_xgb:
        w_cat = 1.0 - w
        blend_pred = w * df_xgb["SalePrice"] + w_cat * df_cat["SalePrice"]
        
        filename = f"submission_005_blend_w{int(w*10):02d}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        submission = pd.DataFrame({
            "Id": df_xgb["Id"],
            "SalePrice": blend_pred
        })
        
        submission.to_csv(filepath, index=False)
        print(f"Generated {filename} (XGB={w:.1f}, Cat={w_cat:.1f})")
        print(f"  -> Range: {blend_pred.min():.2f} - {blend_pred.max():.2f}")

    print("Blending complete.")

if __name__ == "__main__":
    main()
