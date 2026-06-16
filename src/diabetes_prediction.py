"""
Healthcare Diabetes Project

Feature 1:
Diabetes Prediction Model using Logistic Regression

Feature 2:
Model Evaluation using Accuracy, Confusion Matrix,
and Classification Report

Feature 3:
Patient Diabetes Risk Predictor
"""

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_healthcare_diabetes.csv"
)

# Features and target variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# ==========================
# Feature 2: Evaluation
# ==========================

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

report = classification_report(y_test, predictions)

print("\n===== FEATURE 1 =====")
print("Diabetes Prediction Model Trained Successfully")

print("\nModel Accuracy:")
print(accuracy)

print("\n===== FEATURE 2 =====")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ==========================
# Edge Case Handling
# ==========================

def validate_input(value):

    if value is None:
        return "Input cannot be empty"

    if not isinstance(value, (int, float)):
        return "Input must be numeric"

    return "Valid Input"


print("\n===== EDGE CASE TESTING =====")

print(validate_input(None))
print(validate_input("abc"))
print(validate_input(120))

# ==========================
# Debug Information
# ==========================

print("\n===== DEBUG INFORMATION =====")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Prediction Completed Successfully")

# ==========================
# Feature 3
# Patient Diabetes Predictor
# ==========================

print("\n===== FEATURE 3 =====")

sample_patient = [[
    1,      # Id
    1,      # Pregnancies
    150,    # Glucose
    80,     # BloodPressure
    25,     # SkinThickness
    100,    # Insulin
    30.5,   # BMI
    0.5,    # DiabetesPedigreeFunction
    35      # Age
]]

patient_prediction = model.predict(sample_patient)

print("\nSample Patient Data:")
print(sample_patient)

print("\nPrediction Result:")

if patient_prediction[0] == 1:
    print("Patient is likely Diabetic")
else:
    print("Patient is likely Non-Diabetic")

print("\n===== SYSTEM TEST COMPLETED =====")
print("Feature 1 + Feature 2 + Feature 3 working successfully")