"""
Residual learning vs. persistence:
- Target: y = log(volatility_30d + eps) - log(prev_5d_vol + eps)
- Predict y_hat with LightGBM, then reconstruct:
    vol_hat = exp(y_hat) * prev_5d_vol
Time-aware split:
  Train <= 2018-12-31
  Val   2019-01-01 .. 2021-12-31
  Test  >= 2022-01-01
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

EPS = 1e-6

def eval_block(name, y_true, vol_prev5d, logratio_pred):
    """Reconstruct vol prediction and compute metrics."""
    vol_hat = np.exp(logratio_pred) * (vol_prev5d + EPS)
    mae = mean_absolute_error(y_true, vol_hat)
    rmse = np.sqrt(mean_squared_error(y_true, vol_hat))
    r2 = r2_score(y_true, vol_hat)
    print(f"{name} -> R2: {r2:.4f} | MAE: {mae:.5f} | RMSE: {rmse:.5f}")
    return mae, rmse, r2

# Load features 
df = pd.read_csv("features_enriched.csv")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

# Columns
emb_cols = [c for c in df.columns if c.startswith("emb_")]
aux_cols = [
    "prev_5d_vol","prev_5d_absret_mean","prev_5d_ret_std",
    "title_len_chars","title_len_tokens","exclam_count","upper_ratio",
    "tone_neg","tone_neu","tone_pos"
]
for c in aux_cols + ["volatility_30d"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")
use_cols = emb_cols + aux_cols

# Target: log ratio vs persistence
df = df.dropna(subset=["prev_5d_vol","volatility_30d"])
df["y_logratio"] = np.log(df["volatility_30d"] + EPS) - np.log(df["prev_5d_vol"] + EPS)

# Time-aware split 
train_end = pd.Timestamp("2018-12-31")
val_end   = pd.Timestamp("2021-12-31")

train = df[df["date"] <= train_end]
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
test  = df[df["date"] > val_end]

print("Split sizes  Train:", len(train), " Val:", len(val), " Test:", len(test))

# Model 
params = dict(
    n_estimators=900,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
model = lgb.LGBMRegressor(**params)

callbacks = []
try:
    callbacks = [lgb.log_evaluation(50), lgb.early_stopping(120)]
except Exception:
    pass

model.fit(
    train[use_cols], train["y_logratio"],
    eval_set=[(val[use_cols], val["y_logratio"])],
    eval_metric="l2",
    callbacks=callbacks
)

print("\n Residual-model Evaluation (reconstructed vol)")
_ = eval_block("Train", train["volatility_30d"].values,
               train["prev_5d_vol"].values, model.predict(train[use_cols]))
_ = eval_block("Val  ", val["volatility_30d"].values,
               val["prev_5d_vol"].values, model.predict(val[use_cols]))
_ = eval_block("Test ", test["volatility_30d"].values,
               test["prev_5d_vol"].values, model.predict(test[use_cols]))

joblib.dump({"model": model, "use_cols": use_cols, "eps": EPS}, "model_lgbm_residual.pkl")
print("\n Saved model_lgbm_residual.pkl")
