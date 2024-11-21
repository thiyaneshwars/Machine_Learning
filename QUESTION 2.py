# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample DataFrame with monthly electricity consumption data over several years
# Replace this with actual data
data = {
    'year': [2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2022, 2022, 2022],
    'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
    'electricity_consumption': [250, 270, 290, 300, 310, 330, 360, 370, 400, 410, 390, 380, 260, 280, 295]
}

df = pd.DataFrame(data)

# Add a time index to capture the trend (e.g., 1, 2, ..., n for each month in sequence)
df['time_index'] = np.arange(len(df))

# Convert month to a categorical variable and one-hot encode it to capture seasonality
df = pd.get_dummies(df, columns=['month'], drop_first=True)

# Define the features (X) and target variable (y)
X = df.drop(['electricity_consumption', 'year'], axis=1)
y = df['electricity_consumption']

# Split data into training and testing sets
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
