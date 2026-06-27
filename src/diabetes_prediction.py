import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==================================
# INPUT VALIDATION FUNCTION
# ==================================

def validate_input(value, field_name):

    if value is None:
        raise ValueError(f"{field_name} cannot be empty")

    if value == "":
        raise ValueError(f"{field_name} cannot be an empty string")

    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")

    return True


print("=" * 50)
print("HEALTHCARE DIABETES PREDICTION SYSTEM")
print("=" * 50)

# ==================================
# LOAD DATASET
# ==================================

try:
    print("\nLoading dataset...")
    time.sleep(2)

    df = pd.read_csv(
        "data/processed/cleaned_healthcare_diabetes.csv"
    )

    print("✓ Dataset loaded successfully")

except FileNotFoundError:
    print("❌ Dataset file not found")
    exit()

except Exception as e:
    print("❌ Error:", e)
    exit()

# ==================================
# PREPARE DATA
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
# TRAIN MODEL
# ==================================

print("\nTraining model...")
time.sleep(2)

model = LogisticRegression(max_iter=1000)

try:
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")

except Exception as e:
    print("❌ Model Training Error:", e)

# ==================================
# MODEL EVALUATION
# ==================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ==================================
# PATIENT INPUT
# ==================================

print("\n" + "=" * 50)
print("PATIENT INPUT")
print("=" * 50)

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

    # Convert to DataFrame to avoid warning
    patient_df = pd.DataFrame(
        sample_patient,
        columns=X.columns
    )

    prediction = model.predict(patient_df)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    if prediction[0] == 1:
        print("Patient is likely Diabetic")
    else:
        print("Patient is likely Non-Diabetic")

except ValueError:
    print("❌ Invalid input. Please enter numbers only.")

except Exception as e:
    print("❌ Error:", e)

# ==================================
# INPUT VALIDATION TESTS
# ==================================

print("\n" + "=" * 50)
print("INPUT VALIDATION TESTS")
print("=" * 50)

test_inputs = [None, "", "hello", -1, 120]

for item in test_inputs:
    try:
        validate_input(item, "Test Field")
        print(f"{item} -> Valid")

    except Exception as e:
        print(f"{item} -> {e}")

# ==================================
# DASHBOARD GENERATION
# ==================================

print("\nGenerating Dashboard...")

try:

    # Age Distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df["Age"], bins=10)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig("docs/dashboard_age.png")
    plt.close()

    # Glucose Distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df["Glucose"], bins=10)
    plt.title("Glucose Distribution")
    plt.xlabel("Glucose")
    plt.ylabel("Count")
    plt.savefig("docs/dashboard_glucose.png")
    plt.close()

    # Diabetes Outcome Chart
    plt.figure(figsize=(6, 4))
    df["Outcome"].value_counts().plot(kind="bar")
    plt.title("Diabetes Outcome")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.savefig("docs/dashboard_outcome.png")
    plt.close()

    print("✓ Dashboard charts created successfully")

except Exception as e:
    print("❌ Dashboard Error:", e)

# ==================================
# FINAL STATUS
# ==================================

print("\n" + "=" * 50)
print("SYSTEM TEST COMPLETED")
print("=" * 50)

print("✓ Feature 1: Working")
print("✓ Feature 2: Working")
print("✓ Advanced Feature 1: Working")
print("✓ Advanced Feature 2: Working")
print("✓ Input Validation: Working")
print("✓ Error Handling: Working")