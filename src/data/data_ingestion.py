import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def fetch_and_prepare_data(url: str) -> pd.DataFrame:
    """Fetch dataset from URL and preprocess it."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data loaded from {url}, shape: {df.shape}")
        df.drop(columns=['tweet_id'], inplace=True)
        logging.info("'tweet_id' column dropped")
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logging.info("Filtered for 'happiness' and 'sadness' sentiments")
        df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Sentiment labels converted to binary")
        return df
    except Exception as e:
        logging.error(f"Error in fetching/preparing data: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train ({train_data.shape}) and test ({test_data.shape})")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Save train and test data to CSV files."""
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train data saved to {train_path}")
        logging.info(f"Test data saved to {test_path}")
    except Exception as e:
        logging.error(f"Error in saving data: {e}")
        raise

def main(params_path: str = "params.yaml") -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params(params_path)
        test_size = params["data_ingestion"]["test_size"]
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = fetch_and_prepare_data(url)
        train_data, test_data = split_data(df, test_size)
        save_data(train_data, test_data, 'data/raw/train.csv', 'data/raw/test.csv')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()