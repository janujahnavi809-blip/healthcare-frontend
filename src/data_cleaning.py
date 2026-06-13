# Import pandas library
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/Healthcare-Diabetes.csv")

# Display original shape
print("Original Shape:", df.shape)

# Check missing values
print(df.isnull().sum())

# Fill missing numeric values using median
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Check duplicate rows
duplicates_before = df.duplicated().sum()

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_healthcare_diabetes.csv",
    index=False
)

print("Data cleaning completed")