from utils import preprocess_data, evaluate_model
from joblib import load

# Load the new dataset
X_new, y_new = preprocess_data('db/cm1.csv')

# Load the scaler and model
scaler = load('models/scaler.joblib')
svm_model = load('models/svm_model.joblib')

# Standardize the new dataset
X_new = scaler.transform(X_new)

# Predict with the SVM model
y_new_pred = svm_model.predict(X_new)

# Evaluate the model
evaluate_model(y_new, y_new_pred, model=svm_model, X_test=X_new, title="Confusion Matrix for SVM (Testing)")
