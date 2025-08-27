import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data_with_embeddings.csv")
df = df.dropna(subset=["volatility_30d"])

# Select features and target
embedding_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embedding_cols]
y = df["volatility_30d"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to disk
joblib.dump(model, "model_lgbm_volatility.pkl")

print("✅ LightGBM model saved as model_lgbm_volatility.pkl")
