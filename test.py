import pandas as pd
from joblib import load
from utils import preprocess_data, evaluate_model

# Load the test dataset
test_data = pd.read_csv('db/Kc1.csv')

# Preprocess the test dataset
X_test, y_test = preprocess_data(test_data, train=True)  # Ensure 'label' is returned

# Load the scaler and model
model_name = input("Enter the model to test (e.g., RF, SVM, GB, Stacking): ").strip()
model = load(f'models/{model_name}_model.joblib')
scaler = load(f'models/{model_name}_scaler.joblib')

# Scale features
X_test = scaler.transform(X_test)

# Predict and evaluate
evaluate_model(model, X_test, y_test)
