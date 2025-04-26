import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import preprocess_data

def evaluate_model():
    """Evaluate the trained model."""
    # Load model
    model = joblib.load('models/model.pkl')
    
    # Get test data
    _, X_test, _, y_test = preprocess_data()
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\nEvaluation Metrics:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()