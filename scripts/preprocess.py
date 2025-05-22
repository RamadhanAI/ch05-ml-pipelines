import os

# Create the directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Then proceed with your file saving

import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/sales.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()
df.to_csv("data/processed/sales_clean.csv", index=False)
