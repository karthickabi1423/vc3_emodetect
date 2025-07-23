from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json
import os
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    filename='logs/model_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path: str) -> Any:
    """Load the trained model from disk."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load test data and extract features and labels."""
    try:
        test_data = pd.read_csv(test_path)
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        logging.info(f"Test data loaded from {test_path} with shape {test_data.shape}")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Calculate evaluation metrics for the model."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
        logging.info(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logging.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model('models/random_forest_model.pkl')
        X_test, y_test = load_test_data("data/interim/test_bow.csv")
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/evaluation_metrics.json")
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Model evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()