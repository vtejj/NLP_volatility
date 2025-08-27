import gradio as gr
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FinBERT embedding model
finbert = SentenceTransformer('ProsusAI/finbert')

# Load model and data
model = joblib.load("model_lgbm_volatility.pkl")
df = pd.read_csv("data_with_embeddings.csv")

# Extract just the embedding columns
embedding_cols = [col for col in df.columns if col.startswith("emb_")]
example_input = df[embedding_cols].iloc[0].values.tolist()  # use as example

# Prediction function
def predict_volatility(headline):
    # Get embedding
    embedding = finbert.encode(headline).reshape(1, -1)
    # Predict
    prediction = model.predict(embedding)[0]
    return float(prediction)


# Create the Gradio interface
inputs = gr.Textbox(label="Enter News Headline")


demo = gr.Interface(
    fn=predict_volatility,
    inputs=inputs,
    outputs=gr.Number(label="Predicted 30-Day Volatility"),
    title="📰 Market Volatility Predictor from News Headlines",
    description="Paste FinBERT embeddings of a headline to predict the 30-day market volatility. Powered by news sentiment and LightGBM.",
    examples=[example_input]
)

if __name__ == "__main__":
    demo.launch()
