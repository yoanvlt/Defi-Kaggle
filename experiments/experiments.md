# Experiment Tracking

| Run ID | Date | Model | CV MAE (Mean ± Std) | CV MAPE | Description | Submission File | Kaggle Score |
|---|---|---|---|---|---|---|---|
| 001 | 2026-02-12 | HistGradientBoosting | 16918.83 ± 1725.70 | 9.74% | Baseline: SimpleImputer + OneHot + HistGBR | submissions/submission_001_baseline.csv | TBD |
| 002 | 2026-02-12 | XGBoost (Log Target) | 15093.34 ± 1349.12 | - | Log1p Target + XGBRegressor (n_est=5000, lr=0.03) | submissions/submission_002_log_target_boosting.csv | TBD |
| 003 | 2026-02-12 | Tuned XGBoost + FE | 14806.84 ± 1540.52 | 8.53% | Added TotalSF, Age, etc. + RandomizedSearchCV (n_est=4000, lr=0.01) | submissions/submission_003_fe_tuned_xgb.csv | 14125.50704 |
| 004 | 2026-02-12 | CatBoost (Log Target) | 15263.77 ± 1305.28 | 8.61% | Native Cat features + Log1p Target (Depth 6, LR 0.03) | submissions/submission_004_catboost.csv | TBD |
| 005 | 2026-02-12 | Blend (Ensemble) | N/A | N/A | Blend XGB (RUN 003) & Cat (RUN 004). <br> - w=0.9: 14004.83 <br> - w=0.8: 13922.41 <br> - **w=0.7: 13884.08** (Best) <br> - w=0.6: 13899.70 | submissions/submission_005_blend_w07.csv | **13884.08306** (-241.42 vs RUN 003) |
| 006 | 2026-02-12 | Stacking (OOF) | 14346.98 | 8.21% | Stack: XGB(FE) + Cat(Native) + ET. Meta: Ridge. <br> Coefs: XGB~0.48, Cat~0.49, ET~0.07 | submissions/submission_006_stacking.csv | **13868.37275** (Best) |
| 007 | 2026-02-12 | XGB + Poly Features | 14947.58 ± 1256.80 | 8.48% | Added squared terms for areas (TotalSF^2, etc.). Same params as RUN 003. | submissions/submission_007_xgb_poly_area.csv | **13994.69828** (Better than RUN 003) |
| 008 | 2026-02-12 | XGB + Advanced Cleaning | 14050.83 ± 451.85 | 8.11% | Removed outliers (>4000sqft) + Domain-specific imputation (None/0) + Mode imputation. | submissions/submission_008_cleaning_outliers.csv | **13691.83234** (New Best!) |
