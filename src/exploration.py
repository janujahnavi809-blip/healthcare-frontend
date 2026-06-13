# Import pandas library
import pandas as pd

# Load the healthcare diabetes dataset
df = pd.read_csv("data/Healthcare-Diabetes.csv")

# Display dataset shape
print("Shape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Display data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())