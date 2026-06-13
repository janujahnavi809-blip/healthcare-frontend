import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Healthcare-Diabetes.csv")

plt.hist(df['Age'])
plt.title("Age Distribution")
plt.savefig("docs/age_distribution.png")
plt.close()

plt.hist(df['BMI'])
plt.title("BMI Distribution")
plt.savefig("docs/bmi_distribution.png")
plt.close()

plt.hist(df['Glucose'])
plt.title("Glucose Distribution")
plt.savefig("docs/glucose_distribution.png")
plt.close()

print("Charts saved successfully")