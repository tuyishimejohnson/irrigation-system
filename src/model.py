import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data (e.g., handle missing values, encode categorical variables)."""
    df = df.dropna()  # Drop missing values for simplicity
    df = pd.get_dummies(df)  # One-hot encode categorical variables
    return df

def split_data(df, target_column):
    """Split the data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy score."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def save_model(model, scaler, model_path, scaler_path):
    """Save the trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def main(file_path, target_column, model_path, scaler_path):
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    save_model(model, scaler, model_path, scaler_path)
    print(f"Model accuracy: {accuracy}")
