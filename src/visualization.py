# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Healthcare-Diabetes.csv")

# Create Age Distribution chart
plt.hist(df['Age'])
plt.title("Age Distribution")
plt.savefig("docs/age_distribution.png")
plt.close()

# Create BMI Distribution chart
plt.hist(df['BMI'])
plt.title("BMI Distribution")
plt.savefig("docs/bmi_distribution.png")
plt.close()

# Create Glucose Distribution chart
plt.hist(df['Glucose'])
plt.title("Glucose Distribution")
plt.savefig("docs/glucose_distribution.png")
plt.close()

# Display success message
print("Charts saved successfully")