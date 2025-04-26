import os
from pathlib import Path

# Define base paths FIRST
BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Must be defined before use

# Then define file paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'Train_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'Test_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')  # Now MODEL_DIR exists

# Target column (ensure this matches your data)
TRAIN_TARGET_COLUMN = 'class'