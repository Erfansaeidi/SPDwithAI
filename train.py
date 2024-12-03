import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from utils import preprocess_data, evaluate_model

# Define models
MODELS = {
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    # "Stacking": StackingClassifier(estimators=[
    #     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    #     ('svm', SVC(probability=True, random_state=42)),
    #     ('gb', GradientBoostingClassifier(random_state=42))
    # ], final_estimator=GradientBoostingClassifier())
}

# Load dataset
data = pd.read_csv('db/cm1.csv')
X, y = preprocess_data(data)

# Interactive model and SMOTE selection
selected_models = input(f"Available models: {', '.join(MODELS.keys())}. Select models (comma-separated): ").split(',')
use_smote = input("Do you want to apply SMOTE? (yes/no): ").strip().lower() == "yes"

# Train and evaluate each selected model
for model_name in selected_models:
    if model_name not in MODELS:
        print(f"Model {model_name} is not recognized. Skipping...")
        continue

    print(f"\nTraining {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE if selected
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = MODELS[model_name]
    model.fit(X_train, y_train)

    # Save model and scaler
    dump(model, f'models/{model_name}_model.joblib')
    dump(scaler, f'models/{model_name}_scaler.joblib')

    # Evaluate
    evaluate_model(model, X_test, y_test)
