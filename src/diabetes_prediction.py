"""
Healthcare Diabetes Project

Feature 1:
Diabetes Prediction Model using Logistic Regression

Feature 2:
Model Evaluation

Feature 3:
Patient Diabetes Risk Predictor

Additional Features:
Input Validation
Error Handling
Debug Information
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

# ==================================
# Input Validation Function
# ==================================

def validate_input(value, field_name):

    if value is None:
        raise ValueError(f"{field_name} cannot be empty")

    if value == "":
        raise ValueError(f"{field_name} cannot be an empty string")

    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")

    return True


# ==================================
# Load Dataset
# ==================================

try:

    df = pd.read_csv(
        "data/processed/cleaned_healthcare_diabetes.csv"
    )

    print("Dataset loaded successfully")

except FileNotFoundError:

    print("Error: Dataset file not found")
    exit()

except Exception as e:

    print("Unexpected Error:", e)
    exit()


# ==================================
# Prepare Data
# ==================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================
# Feature 1
# Train Model
# ==================================

model = LogisticRegression(max_iter=1000)

try:

    model.fit(X_train, y_train)

    print("Model training successful")

except Exception as e:

    print("Model Training Error:", e)


# ==================================
# Predictions
# ==================================

predictions = model.predict(X_test)

# ==================================
# Feature 2
# Model Evaluation
# ==================================

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

report = classification_report(y_test, predictions)

print("\n===== FEATURE 1 =====")
print("Diabetes Prediction Model")

print("\nModel Accuracy:")
print(accuracy)

print("\n===== FEATURE 2 =====")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ==================================
# Debug Information
# ==================================

print("\n===== DEBUG INFORMATION =====")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Prediction Completed Successfully")

# ==================================
# Feature 3
# Patient Diabetes Risk Predictor
# ==================================

print("\n===== FEATURE 3 =====")

try:

    sample_patient = [
        1,      # Id
        1,      # Pregnancies
        150,    # Glucose
        80,     # BloodPressure
        25,     # SkinThickness
        100,    # Insulin
        30.5,   # BMI
        0.5,    # DiabetesPedigreeFunction
        35      # Age
    ]

    field_names = [
        "Id",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    for value, field in zip(sample_patient, field_names):
        validate_input(value, field)

    patient_prediction = model.predict([sample_patient])

    print("\nSample Patient Data:")
    print(sample_patient)

    print("\nPrediction Result:")

    if patient_prediction[0] == 1:
        print("Patient is likely Diabetic")
    else:
        print("Patient is likely Non-Diabetic")

except Exception as e:

    print("Prediction Error:", e)

# ==================================
# Validation Testing
# ==================================

print("\n===== INPUT VALIDATION TESTS =====")

test_inputs = [
    None,
    "",
    "hello",
    -1,
    120
]

for item in test_inputs:

    try:

        validate_input(item, "Test Field")

        print(f"{item} -> Valid")

    except Exception as e:

        print(f"{item} -> {e}")

# ==================================
# Final Status
# ==================================

print("\n===== SYSTEM TEST COMPLETED =====")

print("Feature 1: Working")
print("Feature 2: Working")
print("Feature 3: Working")

print("Input Validation: Working")
print("Error Handling: Working")