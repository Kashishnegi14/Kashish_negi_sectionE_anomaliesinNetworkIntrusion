import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def detect_intrusion(new_data):
    """Detect intrusion in new network traffic."""
    # Load model and scaler
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Convert input to DataFrame
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Preprocess (same as training)
    categorical_cols = new_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col])
    
    # Scale and predict
    scaled_data = scaler.transform(new_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    # Example test case
    sample_data = {
        'duration': 0,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 215,
        'dst_bytes': 45076,
        # Add all other features from your dataset
    }
    
    pred, prob = detect_intrusion(sample_data)
    status = "Attack" if pred == 1 else "Normal"
    print(f"Prediction: {status} (Confidence: {prob:.2%})")