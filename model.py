from config import MODEL_PATH
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_PATH  # Add this import
import joblib
import os

def create_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        verbose=1,
        n_jobs=-1  # Use all CPU cores
    )

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure dir exists
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")  # Confirmation

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first!")
    return joblib.load(MODEL_PATH)