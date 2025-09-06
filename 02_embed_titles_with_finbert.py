"""
Encoding news headlines using FinBERT from HuggingFace and attach them to the volatility-labeled data.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data 
df = pd.read_csv("data_with_volatility.csv")
titles = df["Title"].astype(str).tolist()

# Step 2: Load FinBERT
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

#  Step 3: Encode Titles
def embed_title(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().cpu().numpy()

embeddings = []
for title in tqdm(titles, desc="Encoding titles"):
    try:
        emb = embed_title(title)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error encoding title: {title[:50]}... Skipped.")
        embeddings.append([0.0] * 768)  # fallback zero vector

# Step 4: Convert to DataFrame
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(768)])
final_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

# Step 5: Save
final_df.to_csv("data_with_embeddings.csv", index=False)
print("✅ Saved to data_with_embeddings.csv")
