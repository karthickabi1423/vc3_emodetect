import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["model_building"]["n_estimators"]
max_depth = params["model_building"]["max_depth"]

# Load the Bag of Words features for training
train_data = pd.read_csv("data/interim/train_bow.csv")

# Separate features and labels
X_train = train_data.drop(columns=['label']).values  # Feature matrix
y_train = train_data['label'].values                # Target vector

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to disk using pickle
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)