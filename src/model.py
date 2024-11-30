from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report
