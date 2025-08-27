import yfinance as yf
import numpy as np

# Download S&P 500 historical data (daily close)
sp500 = yf.download("^GSPC", start="2007-12-01", end="2024-08-25")  # covers headline date range
sp500 = sp500[['Close']].reset_index()

# Compute 5-day log returns and 5-day rolling standard deviation (volatility proxy)
sp500['log_return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['5d_volatility'] = sp500['log_return'].rolling(window=5).std()

# Shift volatility backward so it's the *future* 5-day volatility per date
sp500['target_volatility'] = sp500['5d_volatility'].shift(-5)

# Final columns
sp500 = sp500[['Date', 'target_volatility']]
sp500.columns = ['date', 'target_volatility']  # match headline dataframe

# Preview
print(sp500.head())
