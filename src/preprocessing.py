# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_healthcare_diabetes.csv"
)

# Feature Engineering

# Create BMI category feature
df["BMI_Category"] = (df["BMI"] > 25).astype(int)

# Create High Glucose feature
df["High_Glucose"] = (df["Glucose"] > 140).astype(int)

# Check categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Scale numerical features
scaler = StandardScaler()

# Split dataset into train and test sets
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Save processed files
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing completed")