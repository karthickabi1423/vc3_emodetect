from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json

# Load the trained model from disk
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the test data with Bag of Words features
test_data = pd.read_csv("data/interim/test_bow.csv")

# Extract features and labels from the test data
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted'),
    "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
}

import os
os.makedirs("reports", exist_ok=True)
# Save the evaluation metrics to a JSON file
with open("reports/evaluation_metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)