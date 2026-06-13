# Feature Engineering & Preprocessing Report

## Derived Features Created

1. BMI_Category

   * 1 if BMI > 25
   * 0 otherwise

2. High_Glucose

   * 1 if Glucose > 140
   * 0 otherwise

## Encoding

Dataset contains no categorical variables.

Encoding was not required.

## Feature Scaling

StandardScaler was applied to numerical features.

## Train-Test Split

Training Data: 80%

Testing Data: 20%

Random State: 42

## Output Files

* X_train.csv
* X_test.csv
* y_train.csv
* y_test.csv
