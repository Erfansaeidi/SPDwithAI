import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    if 'defects' in data.columns:
        data['label'] = data['defects'].map({True: 1, False: 0})
    else:
        raise KeyError("The 'defects' column is not found in the dataset!")
    X = data.drop(['label', 'defects'], axis=1)
    y = data['label']
    return X, y

def evaluate_model(y_true, y_pred, model=None, X_test=None, title="Confusion Matrix"):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    if model and X_test is not None:
        print("ROC-AUC Score:", roc_auc_score(y_true, model.decision_function(X_test)))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
