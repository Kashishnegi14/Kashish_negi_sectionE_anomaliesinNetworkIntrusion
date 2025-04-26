# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from config import *

def load_data(include_test_target=False):
    """Load dataset with optional test target handling"""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH) if os.path.exists(TEST_DATA_PATH) else None
    
    # Verify training target exists
    if TRAIN_TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training data missing target column '{TRAIN_TARGET_COLUMN}'")
    
    # Handle test data (which may not have target)
    if include_test_target and (test_df is not None) and (TRAIN_TARGET_COLUMN not in test_df.columns):
        test_df[TRAIN_TARGET_COLUMN] = -1  # Mark missing targets
    
    return train_df, test_df

def preprocess_data(df, is_training=True):
    """Handle feature preprocessing"""
    df = df.copy()
    
    # Convert categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]
    
    # Separate features and target
    if is_training:
        X = df.drop(TRAIN_TARGET_COLUMN, axis=1)
        y = df[TRAIN_TARGET_COLUMN]
        y = y.apply(lambda x: 0 if x == 'normal' else 1)
    else:
        X = df
        y = None
    
    # Scale features
    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(X)
    
    return X, y, scaler