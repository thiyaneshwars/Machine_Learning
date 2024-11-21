# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Simulating a dataset (You can replace this with actual data)
# Assume we have transaction data with features: 'Amount', 'Location', 'User_Behavior', and a binary 'Fraudulent' label
data = {
    'Amount': np.random.rand(1000) * 1000,  # Transaction amount
    'Location': np.random.randint(0, 10, 1000),  # Simplified location (e.g., different regions)
    'User_Behavior': np.random.rand(1000),  # User behavior score (e.g., transaction frequency, patterns)
    'Fraudulent': np.concatenate([np.zeros(950), np.ones(50)])  # 950 non-fraudulent, 50 fraudulent transactions
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the dataset into features (X) and labels (y)
X = df.drop('Fraudulent', axis=1)
y = df['Fraudulent']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (important for many algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementing SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)

# Logistic Regression with class weights adjustment
model = LogisticRegression(class_weight='balanced', random_state=42)

# Create a pipeline that first oversamples the minority class using SMOTE, then trains the logistic regression model
pipeline = Pipeline([
    ('smote', smote),
    ('logreg', model)
])

# Train the model on the training data
pipeline.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test_scaled)

# Evaluate the model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
