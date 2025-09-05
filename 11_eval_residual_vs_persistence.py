"""
Compare residual LGBM vs baselines:
- Model predicts log(vol/prev_5d_vol); reconstruct vol_hat.
- Baselines: constant(train-mean), persistence(prev_5d_vol).
Print MAE/RMSE/R2 and % MAE uplift.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pack = joblib.load("model_lgbm_residual.pkl")
model = pack["model"]; use_cols = pack["use_cols"]; EPS = pack.get("eps", 1e-6)

df = pd.read_csv("features_enriched.csv")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")
df = df.dropna(subset=["prev_5d_vol","volatility_30d"])

train_end = pd.Timestamp("2018-12-31")
val_end   = pd.Timestamp("2021-12-31")
train = df[df["date"] <= train_end]
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
test  = df[df["date"] > val_end]

def reconstruct(vol_prev, logratio_pred):
    return np.exp(logratio_pred) * (vol_prev + EPS)

def eval_block(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_row(tag, mae, rmse, r2, uplift=None):
    up = f"{uplift:.1f}%" if uplift is not None else "-"
    print(f"{tag:<24} MAE={mae:.5f} RMSE={rmse:.5f} R2={r2:.4f} Uplift={up}")

def compare(split_name, split):
    y = split["volatility_30d"].values
    p = split["prev_5d_vol"].values
    # model
    yhat = reconstruct(p, model.predict(split[use_cols]))
    mae_m, rmse_m, r2_m = eval_block(y, yhat)
    # baselines
    mean_train = train["volatility_30d"].mean()
    yhat_const = np.full_like(y, mean_train, dtype=float)
    mae_c, rmse_c, r2_c = eval_block(y, yhat_const)

    yhat_pers = p
    mae_p, rmse_p, r2_p = eval_block(y, yhat_pers)

    # uplifts (MAE reduction)
    up_c = 100.0 * (mae_c - mae_m) / mae_c if mae_c > 0 else 0.0
    up_p = 100.0 * (mae_p - mae_m) / mae_p if mae_p > 0 else 0.0

    print(f"\n=== {split_name} ===")
    print_row("Model (Residual LGBM)", mae_m, rmse_m, r2_m)
    print_row("Baseline: Constant",    mae_c, rmse_c, r2_c, up_c)
    print_row("Baseline: Persistence", mae_p, rmse_p, r2_p, up_p)

compare("Validation (2019–2021)", val)
compare("Test (2022+)", test)
