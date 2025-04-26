from config import SCALER_PATH  # Add this import

def predict():
    try:
        model = load_model()
        _, test_df = load_data(include_test_target=False)
        
        # Ensure scaler exists
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Train first!")
            
        X_test, _, _ = preprocess_data(test_df, is_training=False)
        predictions = model.predict(X_test)
        return ['normal' if x == 0 else 'anomaly' for x in predictions]
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise