import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a simple neural network
# MLPClassifier = Multi-Layer Perceptron Classifier
neural_network = MLPClassifier(
    hidden_layer_sizes=(10, 5),  # Two hidden layers with 10 and 5 neurons
    activation='relu',           # ReLU activation function
    solver='adam',              # Adam optimizer
    max_iter=1000,              # Maximum iterations
    random_state=42
)

# Train the neural network
print("--- Simple Neural Network Example ---")
print("Training neural network...")
neural_network.fit(X_train_scaled, y_train)

# Make predictions
y_pred = neural_network.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Number of layers: {neural_network.n_layers_}")
print(f"Number of iterations: {neural_network.n_iter_}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler, title):
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)

# Plot the results
plot_decision_boundary(X_test, y_test, neural_network, scaler, 
                      'Neural Network Decision Boundary')
# Before plt.savefig, ensure the directory exists
os.makedirs('ai-llm-agent-course/examples/day-06', exist_ok=True)
plt.savefig('ai-llm-agent-course/examples/day-06/neural_network_boundary.png', dpi=150)
plt.show()

# Demonstrate the network's internal structure
print(f"\nNeural Network Architecture:")
print(f"Input layer: {X_train.shape[1]} neurons")
for i, layer_size in enumerate(neural_network.hidden_layer_sizes):
    print(f"Hidden layer {i+1}: {layer_size} neurons")
print(f"Output layer: {len(np.unique(y))} neurons")

# Show how predictions are made
sample_input = X_test_scaled[0:1]  # Take first test sample
prediction = neural_network.predict(sample_input)[0]
probability = neural_network.predict_proba(sample_input)[0]

print(f"\nSample Prediction:")
print(f"Input features: {X_test[0]}")
print(f"Predicted class: {prediction}")
print(f"Prediction probabilities: Class 0: {probability[0]:.3f}, Class 1: {probability[1]:.3f}")

