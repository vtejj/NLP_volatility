# 📈 FinBERT-based S&P 500 Volatility Forecasting

This project uses **daily news headlines** and **FinBERT embeddings** to predict **30-day forward volatility** of the S&P 500 index.  
It demonstrates both the **feasibility** and **challenges** of using language models for financial time-series forecasting.

---

## 🗂️ Dataset

The dataset includes:

- 📅 Date and headline (2008–2024)  
- 💰 S&P 500 daily closing price  
- 📊 Computed 30-day forward volatility (log-return based)  
- 🔤 384-dimensional **FinBERT embeddings** (sentence-transformers)

---

## 🔧 Methodology

### Preprocessing
- Dropped rows with missing volatility  
- Extracted FinBERT embeddings as input features  
- Target: **30-day forward volatility**

### Train-Test Split
- 80% training, 20% testing (`train_test_split`)

### Modeling
Trained 4 models on the headline embeddings:
- 🌲 Random Forest  
- 🚀 XGBoost  
- 💡 LightGBM  
- 🐈 CatBoost  

Evaluated on **R² Score, MAE, RMSE**

---

## 📊 Model Comparison

| Model         | R² Score | MAE    | RMSE   |
|---------------|----------|--------|--------|
| Random Forest | 0.0568   | 0.0024 | 0.0041 |
| XGBoost       | 0.0774   | 0.0024 | 0.0041 |
| **LightGBM**  | **0.0826** | 0.0024 | 0.0041 |
| CatBoost      | 0.0320   | 0.0024 | 0.0042 |

👉 **LightGBM** was chosen for deployment due to its slightly superior R².

---

## 🚀 Gradio Interface

A simple **Gradio demo** allows users to input a news headline and get the model’s predicted 30-day volatility.

**Example:**

- **Headline**: `"Apple beats earnings expectations, shares surge"`  
- **Prediction**: `0.0041` (30-day volatility)

---

## 📝 Key Takeaways
- Combining **NLP embeddings** with financial targets is feasible but challenging.  
- Volatility prediction from text-only inputs shows **modest explanatory power** (R² ~0.08).  
- Highlights the need for **hybrid approaches** (text + market features) for stronger results.  

---
