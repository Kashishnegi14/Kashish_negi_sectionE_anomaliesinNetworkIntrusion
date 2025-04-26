# feature_analysis.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance(model):
    """Extract and visualize feature importance"""
    # Get feature names
    preprocessor = model.named_steps['preprocessor']
    rf = model.named_steps['classifier']
    
    # Get feature names from onehot encoder
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['protocol_type', 'service', 'flag'])
    
    # Combine all feature names
    feature_names = list(preprocessor.named_transformers_['num'].feature_names_in_) + list(categorical_features)
    
    # Get importances
    importances = rf.feature_importances_
    
    # Create DataFrame
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importances

def plot_feature_importance(feature_importances, top_n=20):
    """Plot top N important features"""
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importances.head(top_n),
        palette='viridis'
    )
    plt.title(f'Top {top_n} Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def main():
    # Load trained model
    model = joblib.load('intrusion_detection_model.joblib')
    
    # Get feature importance
    feature_importances = get_feature_importance(model)
    
    print("Top 20 Important Features:")
    print(feature_importances.head(20))
    
    # Plot feature importance
    plot_feature_importance(feature_importances)

if __name__ == "__main__":
    main()