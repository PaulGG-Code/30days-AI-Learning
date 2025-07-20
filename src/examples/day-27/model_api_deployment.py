"""
Day 27 Example: Model API Deployment with FastAPI (Real Model)

This script demonstrates how to deploy a real scikit-learn ML model as a REST API using FastAPI.
The model is trained on the Iris dataset at startup. The API accepts feature vectors and returns predictions.

To run locally:
    pip install fastapi uvicorn scikit-learn
    python examples/day-27/model_api_deployment.py
Then in another terminal:
    uvicorn examples.day-27.model_api_deployment:app --reload

You can then POST to http://127.0.0.1:8000/predict with JSON input, e.g.:
    {"features": [5.1, 3.5, 1.4, 0.2]}

This is a real-world deployment pattern, and can be extended with monitoring, versioning, and MLOps tools.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# --- Train a real model at startup (Iris dataset) ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- FastAPI App ---
app = FastAPI(title="ML Model Deployment Example (Iris Classifier)")

class PredictRequest(BaseModel):
    features: List[float]  # Should be length 4 for Iris

class PredictResponse(BaseModel):
    prediction: int
    class_name: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict endpoint: accepts a list of 4 features and returns a class prediction.
    """
    if len(request.features) != 4:
        return {"prediction": -1, "class_name": "Invalid input: must provide 4 features for Iris dataset."}
    X = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    class_name = iris.target_names[pred]
    return PredictResponse(prediction=pred, class_name=class_name)

# --- MLOps Notes ---
# In production, you would:
# - Load a real trained model (e.g., from disk or a model registry)
# - Add logging and monitoring (e.g., Prometheus, Sentry)
# - Track model and data versions (e.g., MLflow, DVC)
# - Add authentication, rate limiting, and error handling
# - Deploy to cloud (AWS, GCP, Azure) or on-premise infrastructure
# - Set up CI/CD for automated testing and deployment
# - Monitor for model/data drift and trigger retraining as needed 