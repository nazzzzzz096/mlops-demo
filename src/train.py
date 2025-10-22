# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import mlflow

# Load data
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'data.csv')

 # or detect encoding
df = pd.read_csv(data_path)

# Split data
X = df[["feature1", "feature2"]]
y = df["target"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Log metrics with MLflow
mlflow.start_run()
mlflow.log_param("model_type", "DecisionTree")
mlflow.log_metric("accuracy", model.score(X, y))
mlflow.log_artifact("models/model.pkl")
mlflow.end_run()

print("✅ Model trained and logged with MLflow!")




