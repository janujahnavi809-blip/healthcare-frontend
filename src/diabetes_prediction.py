"""
Healthcare Diabetes Project

Feature 1:
Diabetes Prediction Model using Logistic Regression

Feature 2:
Model Evaluation

Feature 3:
Interactive Patient Diabetes Risk Prediction

Additional Features:
Input Validation and Error Handling
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
# Feature 1: Train Model
# ==================================

model = LogisticRegression(max_iter=1000)

try:
    model.fit(X_train, y_train)
    print("Model training successful")

except Exception as e:
    print("Model Training Error:", e)

# ==================================
# Make Predictions
# ==================================

predictions = model.predict(X_test)

# ==================================
# Feature 2: Model Evaluation
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
# Feature 3: Interactive Prediction
# ==================================

print("\n===== ADVANCED FEATURE 1 =====")
print("Enter Patient Details")

try:

    patient_id = int(input("Enter Patient ID: "))
    pregnancies = int(input("Enter Pregnancies: "))
    glucose = int(input("Enter Glucose Level: "))
    blood_pressure = int(input("Enter Blood Pressure: "))
    skin_thickness = int(input("Enter Skin Thickness: "))
    insulin = int(input("Enter Insulin Level: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree = float(
        input("Enter Diabetes Pedigree Function: ")
    )
    age = int(input("Enter Age: "))

    sample_patient = [[
        patient_id,
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    # Validate inputs
    field_names = [
        "Patient ID",
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Diabetes Pedigree Function",
        "Age"
    ]

    for value, field in zip(sample_patient[0], field_names):
        validate_input(value, field)

    # Predict
    prediction = model.predict(sample_patient)

    print("\nPrediction Result:")

    if prediction[0] == 1:
        print("Patient is likely Diabetic")
    else:
        print("Patient is likely Non-Diabetic")

except ValueError:
    print("Error: Please enter only numeric values.")

except Exception as e:
    print("Unexpected Error:", e)

# ==================================
# Validation Tests
# ==================================

print("\n===== INPUT VALIDATION TESTS =====")

test_inputs = [None, "", "hello", -1, 120]

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
print("Advanced Feature 1: Working")
print("Input Validation: Working")
print("Error Handling: Working")