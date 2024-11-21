# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample DataFrame (replace with actual data)
data = {
    'study_hours': [10, 12, 15, 8, 9, 14, 11],
    'attendance_rate': [85, 90, 95, 80, 82, 88, 91],
    'socioeconomic_background': ['low', 'medium', 'high', 'low', 'medium', 'high', 'medium'],
    'test_score': [78, 82, 85, 70, 74, 88, 81]
}

df = pd.DataFrame(data)

# Convert categorical variable to dummy variables
df = pd.get_dummies(df, columns=['socioeconomic_background'], drop_first=True)

# Split data into features (X) and target (y)
X = df.drop('test_score', axis=1)
y = df['test_score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
