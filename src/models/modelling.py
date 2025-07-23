import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import logging
from typing import Tuple
import os

# Configure logging
logging.basicConfig(
    filename='logs/modelling.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data and separate features and labels."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values
        logging.info(f"Loaded training data from {train_path} with shape {train_data.shape}")
        return X_train, y_train
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model to disk using pickle."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main(params_path: str = "params.yaml") -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params(params_path)
        n_estimators = params["model_building"]["n_estimators"]
        max_depth = params["model_building"]["max_depth"]
        X_train, y_train = load_train_data("data/interim/train_bow.csv")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()