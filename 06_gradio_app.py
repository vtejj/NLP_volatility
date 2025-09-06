import gradio as gr
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load residual model (logratio), feature list, and EPS 
pack = joblib.load("model_lgbm_residual.pkl")
model = pack["model"]
use_cols = pack["use_cols"]
EPS = pack.get("eps", 1e-6)

#  Load latest market lag features as proxy 
hist = pd.read_csv("features_enriched.csv", parse_dates=["Date"]).sort_values("Date")
recent = hist.tail(1).iloc[0]  # last row

#  Embedding & tone backbones (FinBERT tone backbone for both)
BERT_NAME = "yiyanghkust/finbert-tone"
tok = AutoTokenizer.from_pretrained(BERT_NAME)
enc = AutoModel.from_pretrained(BERT_NAME).to(DEVICE)
tone = AutoModelForSequenceClassification.from_pretrained(BERT_NAME).to(DEVICE)
enc.eval(); tone.eval()

def embed_cls(text: str):
    x = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out = enc(**x).last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
    return out

def tone_logits(text: str):
    x = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=96).to(DEVICE)
    with torch.no_grad():
        logits = tone(**x).logits.squeeze().detach().cpu().numpy()
    # neg, neu, pos
    return float(logits[0]), float(logits[1]), float(logits[2])

def text_stats(s: str):
    s = s if isinstance(s, str) else ""
    tokens = s.split()
    exclam = s.count("!")
    uppers = sum(1 for c in s if c.isupper())
    letters = sum(1 for c in s if c.isalpha())
    upper_ratio = (uppers / letters) if letters > 0 else 0.0
    return len(s), len(tokens), exclam, upper_ratio

def predict_volatility(headline: str):
    if not headline or not headline.strip():
        return "Please enter a headline."

    row = {}

    # Embeddings: emb_*
    emb = embed_cls(headline)
    for i, v in enumerate(emb):
        row[f"emb_{i}"] = float(v)

    # Tone logits
    n, u, p = tone_logits(headline)
    row["tone_neg"] = n; row["tone_neu"] = u; row["tone_pos"] = p

    # Text stats
    lenc, lent, exclam, upper = text_stats(headline)
    row["title_len_chars"] = lenc
    row["title_len_tokens"] = lent
    row["exclam_count"] = exclam
    row["upper_ratio"] = upper

    # Latest lag features (proxy for “today”)
    row["prev_5d_vol"] = float(recent["prev_5d_vol"])
    row["prev_5d_absret_mean"] = float(recent["prev_5d_absret_mean"])
    row["prev_5d_ret_std"] = float(recent["prev_5d_ret_std"])

    # Align to model’s expected columns
    x = np.array([row.get(c, 0.0) for c in use_cols]).reshape(1, -1)

    # Predict log(vol/prev_5d_vol), reconstruct absolute vol
    logratio = model.predict(x)[0]
    vol_hat = float(np.exp(logratio) * (row["prev_5d_vol"] + EPS))
    return round(vol_hat, 5)

demo = gr.Interface(
    fn=predict_volatility,
    inputs=gr.Textbox(label="Enter market news headline"),
    outputs=gr.Number(label="Predicted 30-Day Volatility"),
    title="News-Driven Volatility Forecast (Residual LGBM + FinBERT)",
    description="Predicts 30-day realized volatility by learning the delta over a persistence baseline from FinBERT embeddings, tone logits, text stats, and recent market lags."
)

if __name__ == "__main__":
    demo.launch()
