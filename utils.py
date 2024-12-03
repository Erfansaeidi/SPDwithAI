import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Preprocess the dataset
def preprocess_data(data, train=True):
    """
    Preprocesses the dataset: handles missing values, creates labels, and splits features/targets.
    
    Parameters:
    - data (pd.DataFrame): The dataset to preprocess.
    - train (bool): Whether this is training data. If False, `y` won't be returned.
    
    Returns:
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series): The labels (if train=True).
    """
    data = data.dropna()

    # Map 'defects' to 'label'
    if 'defects' in data.columns:
        data['label'] = data['defects'].map({True: 1, False: 0})

    # Split features and labels
    X = data.drop(['label', 'defects'], axis=1, errors='ignore')
    y = data['label'] if 'label' in data.columns else None

    if train:
        return X, y
    return X

# Handle SMOTE
def apply_smote(X, y):
    """
    Applies SMOTE to handle class imbalance.
    
    Parameters:
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series): The labels.
    
    Returns:
    - X_resampled, y_resampled: The resampled feature matrix and labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Train a single model
def train_model(X_train, y_train, model, scaler_name, model_name, use_smote=False):
    """
    Trains the model and saves it.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - model: A scikit-learn model instance.
    - scaler_name (str): Name to save the scaler.
    - model_name (str): Name to save the model.
    - use_smote (bool): Whether to apply SMOTE.
    
    Returns:
    - Trained model instance.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    dump(scaler, f'models/{scaler_name}.joblib')  # Save scaler

    # Apply SMOTE if specified
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    # Train model
    model.fit(X_train, y_train)
    dump(model, f'models/{model_name}.joblib')  # Save model
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints metrics.
    
    Parameters:
    - model: The trained model instance.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): True labels for the test set.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Create stacking ensemble
def create_stacking(base_models, final_estimator=None):
    """
    Creates a stacking ensemble.
    
    Parameters:
    - base_models (list): List of (name, model) tuples for base learners.
    - final_estimator: A scikit-learn model instance for meta-learning.
    
    Returns:
    - StackingClassifier instance.
    """
    if final_estimator is None:
        final_estimator = LogisticRegression()
    return StackingClassifier(estimators=base_models, final_estimator=final_estimator)

