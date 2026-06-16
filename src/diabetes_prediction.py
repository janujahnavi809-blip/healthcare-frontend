"""
Feature 1 + Feature 2

Feature 1:
Diabetes Prediction Model using Logistic Regression

Feature 2:
Model Evaluation using Accuracy, Confusion Matrix,
and Classification Report.
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Generate classification report
report = classification_report(y_test, predictions)

# Display results
print("\nModel Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# -----------------------------
# Edge Case Handling
# -----------------------------

def validate_input(value):

    # Check empty input
    if value is None:
        return "Input cannot be empty"

    # Check invalid data type
    if not isinstance(value, (int, float)):
        return "Input must be numeric"

    return "Valid Input"


# Test edge cases
print("\nEdge Case Testing:")

print(validate_input(None))
print(validate_input("abc"))
print(validate_input(120))