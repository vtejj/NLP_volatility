"""
Time-aware training:
- Train: <= 2018-12-31
- Val:   2019-01-01 .. 2021-12-31
- Test:  >= 2022-01-01
Features: emb_* + engineered (lags, text stats, tone logits)
Saves: model_lgbm_enriched.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

def eval_and_print(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> R2: {r2:.4f} | MAE: {mae:.5f} | RMSE: {rmse:.5f}")
    return r2, mae, rmse

# Load & sanitize 
df = pd.read_csv("features_enriched.csv")
df.columns = df.columns.str.strip().str.lower()
if "date" not in df.columns:
    raise ValueError("features_enriched.csv has no 'date' column.")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

emb_cols = [c for c in df.columns if c.startswith("emb_")]
aux_cols = [
    "prev_5d_vol","prev_5d_absret_mean","prev_5d_ret_std",
    "title_len_chars","title_len_tokens","exclam_count","upper_ratio",
    "tone_neg","tone_neu","tone_pos"
]
for c in aux_cols + ["volatility_30d"]:
    if c not in df.columns:
        raise ValueError(f"Missing column in features_enriched.csv: {c}")

use_cols = emb_cols + aux_cols
target = "volatility_30d"

#  Time splits
train_end = pd.Timestamp("2018-12-31")
val_end   = pd.Timestamp("2021-12-31")

train_df = df[df["date"] <= train_end]
val_df   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
test_df  = df[df["date"] > val_end]

print("Split sizes  Train:", len(train_df), " Val:", len(val_df), " Test:", len(test_df))

# model
params = dict(
    n_estimators=700,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    
    verbosity=-1
)

model = lgb.LGBMRegressor(**params)


callbacks = []
try:
    callbacks.append(lgb.log_evaluation(50))      # print every 50 iters
    callbacks.append(lgb.early_stopping(100))     # stop if no val improvement
except Exception:
  
    pass

model.fit(
    train_df[use_cols], train_df[target],
    eval_set=[(val_df[use_cols], val_df[target])],
    eval_metric="l2",
    callbacks=callbacks  
)

print("\nEvaluation")
_ = eval_and_print("Train", train_df[target], model.predict(train_df[use_cols]))
_ = eval_and_print("Val  ", val_df[target],   model.predict(val_df[use_cols]))
_ = eval_and_print("Test ", test_df[target],  model.predict(test_df[use_cols]))

joblib.dump({"model": model, "use_cols": use_cols}, "model_lgbm_enriched.pkl")
print("\nSaved model_lgbm_enriched.pkl")
