import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("data_with_embeddings.csv")

# Drop rows with missing volatility
df = df.dropna(subset=["volatility_30d"])

# Separate features and target
embedding_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embedding_cols]
y = df["volatility_30d"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model_rf_volatility.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # <-- Fix here

print("📊 Evaluation Results:")
print("• R² Score:", round(r2, 4))
print("• MAE:", round(mae, 4))
print("• RMSE:", round(rmse, 4))

