import joblib
import numpy as np

def load_model(file_path):
    """Load a trained model."""
    return joblib.load(file_path)

def predict(model, data):
    """Make predictions on new data."""
    return model.predict(data)
