# Experiment Tracking

| Run ID | Date | Model | CV MAE (Mean ± Std) | CV MAPE | Description | Submission File | Kaggle Score |
|---|---|---|---|---|---|---|---|
| 001 | 2026-02-12 | HistGradientBoosting | 16918.83 ± 1725.70 | 9.74% | Baseline: SimpleImputer + OneHot + HistGBR | submissions/submission_001_baseline.csv | TBD |
| 002 | 2026-02-12 | XGBoost (Log Target) | 15093.34 ± 1349.12 | - | Log1p Target + XGBRegressor (n_est=5000, lr=0.03) | submissions/submission_002_log_target_boosting.csv | TBD |
| 003 | 2026-02-12 | Tuned XGBoost + FE | 14806.84 ± 1540.52 | 8.53% | Added TotalSF, Age, etc. + RandomizedSearchCV (n_est=4000, lr=0.01) | submissions/submission_003_fe_tuned_xgb.csv | TBD |
