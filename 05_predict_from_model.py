import joblib
import pandas as pd
import numpy as np

# Load saved model
model = joblib.load("model_lgbm_volatility.pkl")

# Load data
df = pd.read_csv("data_with_embeddings.csv")
df = df.dropna(subset=["volatility_30d"])

# Select a few samples to predict
embedding_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embedding_cols]

# Predict on first 5 samples
sample = X.head(5)
preds = model.predict(sample)

# Display results
for i, p in enumerate(preds):
    print(f"Sample {i+1} → Predicted 30-day volatility: {round(p, 5)}")
