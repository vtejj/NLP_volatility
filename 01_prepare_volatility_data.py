"""
Prepare data for volatility prediction from news headlines and S&P500 prices.
"""

import pandas as pd
import numpy as np

# Step 1: Load Data 
df = pd.read_csv("data/sp500_headlines_2008_2024.csv")

# Ensure date is datetime and sorted
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Step 2: Compute Log Returns 
df['log_return'] = np.log(df['CP'] / df['CP'].shift(1))

# Step 3: Compute 30-day Rolling Volatility 
df['volatility_30d'] = df['log_return'].rolling(window=30).std()

# Step 4: Drop NaNs (from shifting and rolling) 
df = df.dropna(subset=['log_return', 'volatility_30d'])

#  Step 5: Final Sanity Check 
print(df[['Date', 'CP', 'log_return', 'volatility_30d', 'Title']].head())

# Save preprocessed data 
df.to_csv("data_with_volatility.csv", index=False)
