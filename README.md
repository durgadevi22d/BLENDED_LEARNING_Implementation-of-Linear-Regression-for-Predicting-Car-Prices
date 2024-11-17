# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:
Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset:
Load the dataset containing car prices and relevant features.
3. Data Preprocessing:
Handle missing values and perform feature selection if necessary.
4. Split Data:
Split the dataset into training and testing sets.
5. Train Model:
Create a linear regression model and fit it to the training data.
6. Make Predictions:
Use the model to make predictions on the test set.
7. Evaluate Model:
Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions:
Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results:
Display the predictions and evaluation metrics.


## Program:
```
Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: DURGADEVI P
RegisterNumber: 212223100006 

# Import necessary libraries
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import shapiro
import numpy as np

# Load the dataset from the file path
data = pd.read_csv(r"C:\Users\admin\Downloads\CarPrice_Assignment (1).csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Data Preprocessing
# Handle missing values (if any)
data = data.dropna()  # Drop rows with missing values

# Select features and target variable
# Assume 'price' is the target variable and 'horsepower', 'curbweight', 'enginesize', and 'highwaympg' are features
X = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Assumption 1: Linearity
# Check residuals vs predicted values for linearity
plt.scatter(y_pred, y_test - y_pred, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Linearity Check)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Assumption 2: Homoscedasticity
# Homoscedasticity means the residuals should have constant variance.
plt.scatter(y_pred, y_test - y_pred, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Homoscedasticity Check')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Assumption 3: Normality
# Use the Shapiro-Wilk test for normality of residuals
residuals = y_test - y_pred
stat, p_value = shapiro(residuals)
print("Shapiro-Wilk Test:")
print(f"Statistic: {stat}, P-value: {p_value}")
if p_value > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")

# Assumption 4: Multicollinearity
# Check Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_with_const = add_constant(X)  # Add a constant column for VIF calculation
vif_data = pd.DataFrame({
    "Feature": X_with_const.columns,
    "VIF": [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
})
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Plotting Residual Distribution for Normality
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![1o1](https://github.com/user-attachments/assets/f8f5b668-65e9-49d0-baad-ad7c87853e6f)
![102](https://github.com/user-attachments/assets/93a08063-7977-4de9-af5e-b913df8259b5)
![103](https://github.com/user-attachments/assets/3f34aa71-c94a-41ce-a34b-ba0aba38173b)
![104](https://github.com/user-attachments/assets/9e42fc96-dea0-41d4-bbc9-aa1e38b5c1b9)
![105](https://github.com/user-attachments/assets/07a941f3-c5b5-4c33-b56c-279bec5df507)



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
