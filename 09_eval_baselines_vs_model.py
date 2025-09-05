"""
Evaluate LightGBM vs naive baselines and report MAE uplift:
- Constant baseline: predict train mean volatility
- Persistence baseline: predict prev_5d_vol
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model + features 
pack = joblib.load("model_lgbm_enriched.pkl")
model = pack["model"]
use_cols = pack["use_cols"]
target = "volatility_30d"

df = pd.read_csv("features_enriched.csv", parse_dates=["Date"]).sort_values("Date")

# Time-aware splits 
train_end = pd.Timestamp("2018-12-31")
val_end   = pd.Timestamp("2021-12-31")

train_df = df[df["Date"] <= train_end].copy()
val_df   = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
test_df  = df[df["Date"] > val_end].copy()

def eval_reg(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_row(name, mae, rmse, r2, uplift=None):
    up = f"{uplift:.1f}%" if uplift is not None else "-"
    print(f"{name:<22}  MAE={mae:.5f}  RMSE={rmse:.5f}  R2={r2:.4f}  Uplift_vs_baseline={up}")

print(f"Split sizes  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

#  MODEL on VAL/TEST 
y_val  = val_df[target].values
y_test = test_df[target].values

yhat_val_model  = model.predict(val_df[use_cols])
yhat_test_model = model.predict(test_df[use_cols])

mae_val_m, rmse_val_m, r2_val_m   = eval_reg(y_val,  yhat_val_model)
mae_test_m, rmse_test_m, r2_test_m = eval_reg(y_test, yhat_test_model)

#  BASELINE 1: Constant (train mean) 
train_mean = train_df[target].mean()
yhat_val_const  = np.full_like(y_val,  train_mean, dtype=float)
yhat_test_const = np.full_like(y_test, train_mean, dtype=float)

mae_val_c, rmse_val_c, r2_val_c   = eval_reg(y_val,  yhat_val_const)
mae_test_c, rmse_test_c, r2_test_c = eval_reg(y_test, yhat_test_const)

#  BASELINE 2: Persistence (prev_5d_vol) 
# If any NaNs, drop those rows for fair comparison
def persistence_eval(split_df):
    y_true = split_df[target].values
    y_pred = split_df["prev_5d_vol"].values
    mask = ~np.isnan(y_pred)
    return eval_reg(y_true[mask], y_pred[mask]), mask.sum(), mask.size

(val_p_metrics, n_ok_val, n_all_val)   = persistence_eval(val_df)
(test_p_metrics, n_ok_test, n_all_test) = persistence_eval(test_df)
mae_val_p, rmse_val_p, r2_val_p   = val_p_metrics
mae_test_p, rmse_test_p, r2_test_p = test_p_metrics

#  UPLIFT (% MAE reduction) 
uplift_val_const  = 100.0 * (mae_val_c  - mae_val_m)  / mae_val_c  if mae_val_c  > 0 else 0.0
uplift_test_const = 100.0 * (mae_test_c - mae_test_m) / mae_test_c if mae_test_c > 0 else 0.0

uplift_val_p  = 100.0 * (mae_val_p  - mae_val_m)  / mae_val_p  if mae_val_p  > 0 else 0.0
uplift_test_p = 100.0 * (mae_test_p - mae_test_m) / mae_test_p if mae_test_p > 0 else 0.0

#  Print table 
print("\nValidation (2019–2021) ")
print_row("Model (LightGBM)", mae_val_m, rmse_val_m, r2_val_m)
print_row("Baseline: Constant", mae_val_c, rmse_val_c, r2_val_c,
          uplift=uplift_val_const)
print_row(f"Baseline: Persistence", mae_val_p, rmse_val_p, r2_val_p,
          uplift=uplift_val_p)

print("\n Test (2022+) ")
print_row("Model (LightGBM)", mae_test_m, rmse_test_m, r2_test_m)
print_row("Baseline: Constant", mae_test_c, rmse_test_c, r2_test_c,
          uplift=uplift_test_const)
print_row(f"Baseline: Persistence", mae_test_p, rmse_test_p, r2_test_p,
          uplift=uplift_test_p)

#save to csv
out = pd.DataFrame([
    {"split":"val","name":"model_lgbm","mae":mae_val_m,"rmse":rmse_val_m,"r2":r2_val_m,"uplift_vs_const(%)":uplift_val_const,"uplift_vs_persist(%)":uplift_val_p},
    {"split":"val","name":"baseline_constant","mae":mae_val_c,"rmse":rmse_val_c,"r2":r2_val_c},
    {"split":"val","name":"baseline_persistence","mae":mae_val_p,"rmse":rmse_val_p,"r2":r2_val_p},
    {"split":"test","name":"model_lgbm","mae":mae_test_m,"rmse":rmse_test_m,"r2":r2_test_m,"uplift_vs_const(%)":uplift_test_const,"uplift_vs_persist(%)":uplift_test_p},
    {"split":"test","name":"baseline_constant","mae":mae_test_c,"rmse":rmse_test_c,"r2":r2_test_c},
    {"split":"test","name":"baseline_persistence","mae":mae_test_p,"rmse":mae_test_p,"r2":r2_test_p},
])
out.to_csv("baseline_vs_model_metrics.csv", index=False)
print("\nSaved baseline_vs_model_metrics.csv")
