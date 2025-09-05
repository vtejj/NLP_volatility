"""
Enriching features for volatility prediction:
- Price lags (prev_5d_vol, prev_5d_absret_mean, prev_5d_ret_std)
- Text stats (title_len_chars, title_len_tokens, exclam_count, upper_ratio)
- FinBERT tone logits (neg/neu/pos)
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TONE_MODEL = "yiyanghkust/finbert-tone"

#  Load base data 
base = pd.read_csv("data_with_volatility.csv")
base["Date"] = pd.to_datetime(base["Date"])
base = base.sort_values("Date").reset_index(drop=True)
# Ensure required cols exist
req_cols = {"Date","CP","Title","volatility_30d"}
missing = req_cols - set(base.columns)
if missing:
    raise ValueError(f"Missing columns in data_with_volatility.csv: {missing}")

#  Price-derived lag features (shifted to avoid leakage) 
base["log_return"] = np.log(base["CP"] / base["CP"].shift(1))
base["prev_5d_vol"] = base["log_return"].rolling(5, min_periods=5).std().shift(1)
base["prev_5d_absret_mean"] = base["log_return"].abs().rolling(5, min_periods=5).mean().shift(1)
base["prev_5d_ret_std"] = base["log_return"].rolling(5, min_periods=5).std().shift(1)

#  Simple text stats 
def text_stats(s: str):
    s = s if isinstance(s, str) else ""
    tokens = s.split()
    exclam = s.count("!")
    uppers = sum(1 for c in s if c.isupper())
    letters = sum(1 for c in s if c.isalpha())
    upper_ratio = (uppers / letters) if letters > 0 else 0.0
    return len(s), len(tokens), exclam, upper_ratio

tstats = base["Title"].apply(text_stats)
base["title_len_chars"] = [t[0] for t in tstats]
base["title_len_tokens"] = [t[1] for t in tstats]
base["exclam_count"] = [t[2] for t in tstats]
base["upper_ratio"] = [t[3] for t in tstats]

#  FinBERT tone logits (neg, neu, pos) 
tokenizer = AutoTokenizer.from_pretrained(TONE_MODEL)
tone_model = AutoModelForSequenceClassification.from_pretrained(TONE_MODEL).to(DEVICE)
tone_model.eval()

def finbert_logits(text: str):
    text = text if isinstance(text, str) else ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96).to(DEVICE)
    with torch.no_grad():
        logits = tone_model(**inputs).logits.squeeze().detach().cpu().numpy()
    return float(logits[0]), float(logits[1]), float(logits[2])  # neg, neu, pos

neg_list, neu_list, pos_list = [], [], []
for t in tqdm(base["Title"], desc="FinBERT tone"):
    try:
        n, u, p = finbert_logits(t)
    except Exception:
        n, u, p = 0.0, 0.0, 0.0
    neg_list.append(n); neu_list.append(u); pos_list.append(p)

base["tone_neg"] = neg_list
base["tone_neu"] = neu_list
base["tone_pos"] = pos_list

#  Merge with existing embeddings (emb_*) 
emb = pd.read_csv("data_with_embeddings.csv")
emb["Date"] = pd.to_datetime(emb["Date"])
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
if not emb_cols:
    raise ValueError("No emb_* columns found in data_with_embeddings.csv. Re-run 02_embed_titles_with_finbert.py")

# Keep only date, target, and embeddings; inner-join ensures row alignment
emb_slim = emb[["Date","volatility_30d"] + emb_cols].dropna(subset=["volatility_30d"]).copy()
feat = pd.merge(
    base,
    emb_slim,
    on=["Date","volatility_30d"],
    how="inner"
)

# Final clean: drop rows where any lag features are NA
feat = feat.dropna(subset=[
    "prev_5d_vol","prev_5d_absret_mean","prev_5d_ret_std",
    "title_len_chars","title_len_tokens","exclam_count","upper_ratio",
    "tone_neg","tone_neu","tone_pos"
])

feat.to_csv("features_enriched.csv", index=False)
print(" Saved features_enriched.csv with shape:", feat.shape)
