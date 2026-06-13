import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_healthcare_diabetes.csv")

print("Original Shape:", df.shape)

# ==========================
# Feature Engineering
# ==========================

# Feature 1: BMI Category
df["BMI_Category"] = (df["BMI"] > 25).astype(int)

# Feature 2: High Glucose Flag
df["High_Glucose"] = (df["Glucose"] > 140).astype(int)

print("New Features Created")

# ==========================
# Encoding
# ==========================

# Check for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Categorical Columns:")
print(list(categorical_cols))

# No categorical columns found
# Encoding not required

# ==========================
# Feature Scaling
# ==========================

numerical_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

scaler = StandardScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Scaling Complete")

# ==========================
# Train Test Split
# ==========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Files Saved Successfully")