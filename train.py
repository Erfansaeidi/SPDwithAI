import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from joblib import dump
from collections import Counter
from utils import preprocess_data

# Load and preprocess the dataset
X, y = preprocess_data('db/KC1.csv')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
dump(scaler, 'models/scaler.joblib')

# Check class imbalance
print("Label distribution in training set before SMOTE:", Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Label distribution in training set after SMOTE:", Counter(y_train))

# Train SVM Classifier
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save the trained model
dump(svm_model, 'models/svm_model.joblib')

# Evaluate on the test set
y_pred = svm_model.predict(X_test)
from utils import evaluate_model
evaluate_model(y_test, y_pred, model=svm_model, X_test=X_test, title="Confusion Matrix for SVM (Training)")
