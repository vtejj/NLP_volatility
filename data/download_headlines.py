import pandas as pd

# Load the CSV
df = pd.read_csv("data/sp500_headlines_2008_2024.csv")

# Convert 'date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date (just in case)
df = df.sort_values('Date')

# View first few rows
print(df.head())