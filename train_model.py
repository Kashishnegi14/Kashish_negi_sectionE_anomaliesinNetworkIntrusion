from preprocessing import load_data, preprocess_data
from model import create_model, save_model
from config import MODEL_PATH, SCALER_PATH  # Add SCALER_PATH import
import os

def train():
    print("Starting training...")
    os.makedirs('models', exist_ok=True)  # Ensure directory exists
    
    try:
        # Load and preprocess data
        train_df, _ = load_data(include_test_target=False)
        X_train, y_train, _ = preprocess_data(train_df, is_training=True)
        
        # Train model
        model = create_model()
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Save artifacts
        save_model(model)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()