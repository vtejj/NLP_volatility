import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# Load data
df = pd.read_csv("data_with_embeddings.csv")
df = df.dropna(subset=["volatility_30d"])

# Features and target
embedding_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embedding_cols]
y = df["volatility_30d"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM
print("Training LightGBM...")
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgbm_model.fit(X_train, y_train)
joblib.dump(lgbm_model, "model_lgbm_volatility.pkl")

y_pred_lgbm = lgbm_model.predict(X_test)
print("\nLightGBM Results:")
print("R² Score:", round(r2_score(y_test, y_pred_lgbm), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred_lgbm), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_lgbm)), 4))

# CatBoost
print("\nTraining CatBoost...")
cat_model = CatBoostRegressor(verbose=0, iterations=100, random_state=42)
cat_model.fit(X_train, y_train)
joblib.dump(cat_model, "model_catboost_volatility.pkl")

y_pred_cat = cat_model.predict(X_test)
print("\nCatBoost Results:")
print("R² Score:", round(r2_score(y_test, y_pred_cat), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred_cat), 4))
print("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred_cat)))  # <-- Fix here

