# FinBERT-based S&P 500 Volatility Forecasting

This project uses **daily news headlines** and **FinBERT embeddings** to predict **30-day forward volatility** of the S&P 500 index.  
It demonstrates both the **feasibility** and **challenges** of using language models for financial time-series forecasting.

---

##  Dataset

The dataset includes:

- Date and headline (2008–2024)  
- S&P 500 daily closing price  
- Computed 30-day forward volatility (log-return based)  
- 384-dimensional **FinBERT embeddings** (sentence-transformers)

---

## Methodology

### Preprocessing
- Dropped rows with missing volatility  
- Extracted FinBERT embeddings as input features  
- Target: **30-day forward volatility**

### Train-Test Split
- 80% training, 20% testing (`train_test_split`)

### Modeling
Trained 4 models on the headline embeddings:
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

Evaluated on **R² Score, MAE, RMSE**

---

## Model Comparison

| Model         | R² Score | MAE    | RMSE   |
|---------------|----------|--------|--------|
| Random Forest | 0.0568   | 0.0024 | 0.0041 |
| XGBoost       | 0.0774   | 0.0024 | 0.0041 |
| **LightGBM**  | **0.0826** | 0.0024 | 0.0041 |
| CatBoost      | 0.0320   | 0.0024 | 0.0042 |

 **LightGBM** was chosen for deployment due to its slightly superior R².

---
## Baselines & MAE Uplift

We compare against two naive baselines:
- **Constant**: predict the train-mean volatility every day  
- **Persistence**: predict **prev_5d_vol** (yesterday’s 5-day rolling vol)

| Split            | Model / Baseline        | MAE     | RMSE    | R²      | MAE Uplift vs Baseline |
|------------------|-------------------------|---------|---------|---------|------------------------|
| 2019–2021 (Val)  | **Residual LGBM **| 0.00264 | 0.00494 | -0.0325 | —                      |
| 2019–2021 (Val)  | Constant (train-mean)   | 0.00339 | 0.00490 | -0.0179 | **+22.1%**             |
| 2019–2021 (Val)  | Persistence (prev_5d)   | 0.00341 | 0.00601 | -0.5307 | **+22.5%**             |
| 2022+ (Test)     | **Residual LGBM **| 0.00165 | 0.00202 | -0.3287 | —                      |
| 2022+ (Test)     | Constant (train-mean)   | 0.00330 | 0.00354 | -3.0985 | **+50.1%**             |
| 2022+ (Test)     | Persistence (prev_5d)   | 0.00187 | 0.00253 | -1.0860 | **+12.0%**             |

> **Takeaway:** The residual LightGBM consistently **reduces MAE** vs strong naive baselines, including a **12% MAE improvement vs persistence** on the 2022+ test regime.


## Gradio Interface

A simple **Gradio demo** allows users to input a news headline and get the model’s predicted 30-day volatility.

**Example:**

- **Headline**: `"Apple beats earnings expectations, shares surge"`  
- **Prediction**: `0.0041` (30-day volatility)

---


