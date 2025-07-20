import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Generate sample data for house prices
np.random.seed(42)
house_sizes = np.random.uniform(500, 3000, 100)  # House sizes in sq ft
# Price = base price + price per sq ft + some noise
house_prices = 50000 + 100 * house_sizes + np.random.normal(0, 10000, 100)

# Reshape for sklearn (needs 2D array)
X = house_sizes.reshape(-1, 1)
y = house_prices

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Supervised Learning Example: House Price Prediction ---")
print(f"Model trained on {len(X_train)} houses")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Model equation: Price = ${model.intercept_:,.2f} + ${model.coef_[0]:.2f} * Size")

# Visualize the results
os.makedirs('ai-llm-agent-course/examples/day-03', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Prices')
plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ai-llm-agent-course/examples/day-03/house_price_prediction.png', dpi=150)
plt.show()

# Example prediction for a new house
new_house_size = 2000
predicted_price = model.predict([[new_house_size]])[0]
print(f"\nPrediction for a {new_house_size} sq ft house: ${predicted_price:,.2f}")

