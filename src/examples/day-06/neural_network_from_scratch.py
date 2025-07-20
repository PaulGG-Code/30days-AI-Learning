"""
Day 6 Advanced Example: Neural Network from Scratch
This script builds a simple neural network (1 hidden layer) from scratch using numpy for binary classification.
It demonstrates forward and backward propagation, and compares results to scikit-learn's MLPClassifier.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0)
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Generate a toy dataset (moons)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network parameters
input_dim = X_train.shape[1]
hidden_dim = 8
output_dim = 1
learning_rate = 0.1
epochs = 2000

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

losses = []
# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    # Loss
    loss = binary_cross_entropy(y_train.reshape(-1, 1), a2)
    losses.append(loss)
    # Backward pass
    dz2 = a2 - y_train.reshape(-1, 1)
    dW2 = a1.T @ dz2 / len(X_train)
    db2 = np.mean(dz2, axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X_train.T @ dz1 / len(X_train)
    db1 = np.mean(dz1, axis=0, keepdims=True)
    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Plot loss curve
plt.figure(figsize=(7,4))
plt.plot(losses)
plt.title('Loss Curve (Neural Network from Scratch)')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Evaluate on test set
z1_test = X_test @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = sigmoid(z2_test)
y_pred = (a2_test > 0.5).astype(int).flatten()
acc = accuracy_score(y_test, y_pred)
print(f"\nNeural Network from Scratch Test Accuracy: {acc:.3f}")

# Compare to scikit-learn's MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)
sk_acc = accuracy_score(y_test, mlp.predict(X_test))
print(f"scikit-learn MLPClassifier Test Accuracy: {sk_acc:.3f}")

# Decision boundary visualization
def plot_decision_boundary(model_func, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_func(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

# Plot decision boundary for custom NN
plot_decision_boundary(lambda X_: (sigmoid(relu(X_ @ W1 + b1) @ W2 + b2) > 0.5).astype(int).flatten(), X_test, y_test, 'Decision Boundary: NN from Scratch')
# Plot decision boundary for scikit-learn MLP
plot_decision_boundary(lambda X_: mlp.predict(X_), X_test, y_test, 'Decision Boundary: scikit-learn MLPClassifier') 