import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import os

# Generate sample customer data
np.random.seed(42)
n_customers = 300

# Create synthetic customer data with natural clusters
X, _ = make_blobs(n_samples=n_customers, centers=3, n_features=2, 
                  random_state=42, cluster_std=1.5)

# Let's say the features are: [Annual Spending, Visit Frequency]
annual_spending = X[:, 0] * 1000 + 5000  # Scale to realistic spending amounts
visit_frequency = X[:, 1] * 2 + 10       # Scale to realistic visit frequency

# Combine into our dataset
customer_data = np.column_stack([annual_spending, visit_frequency])

# Apply K-Means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(customer_data)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

print("--- Unsupervised Learning Example: Customer Segmentation ---")
print(f"Analyzed {n_customers} customers")
print(f"Identified {k} customer segments")

# Analyze each cluster
for i in range(k):
    cluster_customers = customer_data[cluster_labels == i]
    avg_spending = np.mean(cluster_customers[:, 0])
    avg_visits = np.mean(cluster_customers[:, 1])
    cluster_size = len(cluster_customers)
    
    print(f"\nSegment {i+1} ({cluster_size} customers):")
    print(f"  Average Annual Spending: ${avg_spending:,.2f}")
    print(f"  Average Visit Frequency: {avg_visits:.1f} visits/month")
    
    # Assign segment names based on characteristics
    if avg_spending > 7000 and avg_visits > 12:
        segment_name = "VIP Customers"
    elif avg_spending > 5000:
        segment_name = "Regular Customers"
    else:
        segment_name = "Occasional Customers"
    print(f"  Segment Type: {segment_name}")

# Visualize the clustering results
plt.figure(figsize=(12, 5))

# Plot 1: Original data
plt.subplot(1, 2, 1)
plt.scatter(customer_data[:, 0], customer_data[:, 1], alpha=0.6, color='gray')
plt.xlabel('Annual Spending ($)')
plt.ylabel('Visit Frequency (per month)')
plt.title('Customer Data (Before Clustering)')
plt.grid(True, alpha=0.3)

# Plot 2: Clustered data
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(k):
    cluster_customers = customer_data[cluster_labels == i]
    plt.scatter(cluster_customers[:, 0], cluster_customers[:, 1], 
                c=colors[i], alpha=0.6, label=f'Segment {i+1}')

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
            c='black', marker='x', s=200, linewidths=3, label='Centroids')

plt.xlabel('Annual Spending ($)')
plt.ylabel('Visit Frequency (per month)')
plt.title('Customer Segmentation (After Clustering)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Before plt.savefig, ensure the directory exists
os.makedirs('ai-llm-agent-course/examples/day-04', exist_ok=True)
plt.savefig('ai-llm-agent-course/examples/day-04/customer_segmentation.png', dpi=150)
plt.show()

# Demonstrate how to classify a new customer
new_customer_spending = 6500
new_customer_visits = 8
new_customer = np.array([[new_customer_spending, new_customer_visits]])
predicted_segment = kmeans.predict(new_customer)[0]

print(f"\nNew Customer Classification:")
print(f"Spending: ${new_customer_spending:,}, Visits: {new_customer_visits}/month")
print(f"Predicted Segment: {predicted_segment + 1}")

