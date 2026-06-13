import pandas as pd

# Load dataset
df = pd.read_csv("data/Healthcare-Diabetes.csv")

print("Original Shape:", df.shape)

# -------------------------
# Missing Values
# -------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with median
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# -------------------------
# Remove Duplicates
# -------------------------
duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates_before)

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

# -------------------------
# Data Types
# -------------------------
print("\nData Types:")
print(df.dtypes)

# -------------------------
# Save Clean Dataset
# -------------------------
output_path = "data/processed/cleaned_healthcare_diabetes.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully.")
print("Final Shape:", df.shape)
print("Saved to:", output_path)