import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from joblib import dump
#FOR SVM
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('db/KC1.csv')
#data, meta = arff.loadarff('db/pc2.arff')
# Convert to DataFrame
#df = pd.DataFrame(data)
# Save to CSV
#df.to_csv('your_dataset.csv', index=False)


print("First 5 rows of the dataset:\n", data.head())



# Inspect dataset
print(data.info())
print(data.describe())
print("Missing values:\n", data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Map the 'defects' column to 'label'
if 'defects' in data.columns:
    data['label'] = data['defects'].map({True: 1, False: 0})
else:
    raise KeyError("The 'defects' column is not found in the dataset!")

# Verify mapping
print("First 5 rows after label mapping:\n", data[['defects', 'label']].head())

# Define features and target
X = data.drop(['label', 'defects'], axis=1)  # Drop both 'label' and 'defects'
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check class imbalance
print("Label distribution in training set before SMOTE:", Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Label distribution in training set after SMOTE:", Counter(y_train))

# Train Random Forest Classifier
#rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_model.fit(X_train, y_train)

# Train SVM Classifier
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predictions
#y_pred = rf_model.predict(X_test)
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Additional metrics
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1]))


# Save the scaler after fitting it during training
dump(scaler, 'scaler.joblib')



#test dataset
# Load the new dataset
new_data = pd.read_csv('db/cm1.csv')

# Preprocess the new dataset (similar steps to the original data)
new_data = new_data.dropna()
if 'defects' in new_data.columns:
    new_data['label'] = new_data['defects'].map({True: 1, False: 0})
X_new = new_data.drop(['label', 'defects'], axis=1)
y_new = new_data['label']

# Load the scaler and model (previously saved during training)
scaler = load('scaler.joblib')  # The scaler fit during training
svm_model = load('svm_model.joblib')  # The trained SVM model

# Standardize the new dataset
X_new = scaler.transform(X_new)

# Predict with the SVM model
y_new_pred = svm_model.predict(X_new)

# Evaluate the model
print("New Dataset Accuracy:", accuracy_score(y_new, y_new_pred))
print("New Dataset F1 Score:", f1_score(y_new, y_new_pred))
print("New Dataset ROC-AUC Score:", roc_auc_score(y_new, svm_model.decision_function(X_new)))
print("New Dataset Classification Report:\n", classification_report(y_new, y_new_pred))

# Confusion Matrix
cm = confusion_matrix(y_new, y_new_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM')
plt.show()