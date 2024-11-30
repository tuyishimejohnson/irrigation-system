import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load the dataset."""
    file_path = "./notebook/cropdata_updated.csv"
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """Preprocess the dataset."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
