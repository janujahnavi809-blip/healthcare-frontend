import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "data/processed/cleaned_healthcare_diabetes.csv"
)

# Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("docs/dashboard_age.png")
plt.close()

# Glucose Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Glucose"], bins=10)
plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Count")
plt.savefig("docs/dashboard_glucose.png")
plt.close()

# Diabetes Outcome Count
plt.figure(figsize=(6,4))
df["Outcome"].value_counts().plot(kind="bar")
plt.title("Diabetes Outcome")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.savefig("docs/dashboard_outcome.png")
plt.close()

print("Dashboard charts created successfully!")