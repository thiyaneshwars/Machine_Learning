import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize variables for cross-validation
k_values = range(1, 21)
mean_scores = []

# Cross-validation for different values of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_scores.append(scores.mean())

# Find optimal k
optimal_k = k_values[np.argmax(mean_scores)]
print(f'Optimal value of k: {optimal_k}')

# Train final model with optimal k
final_model = KNeighborsClassifier(n_neighbors=optimal_k)
final_model.fit(X_train, y_train)

# Evaluate on test set
test_accuracy = final_model.score(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')
