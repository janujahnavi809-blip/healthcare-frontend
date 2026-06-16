"""
Feature: Diabetes Prediction Model

This feature trains a Logistic Regression model
to predict whether a patient has diabetes based
on healthcare information such as Glucose, BMI,
Blood Pressure, Insulin, and Age.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_healthcare_diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
# Function to validate user input
def validate_input(value):

    # Check for empty input
    if value is None:
        return "Input cannot be empty"

    # Check for wrong data type
    if not isinstance(value, (int, float)):
        return "Input must be numeric"

    return "Valid Input"

# Test cases
print(validate_input(None))
print(validate_input("abc"))
print(validate_input(120))