import pandas as pd
import os

# Load raw data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# Example: just copy to processed (replace with your real preprocessing)
os.makedirs('data/processed', exist_ok=True)
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)