import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../../data/flowers_dataset_cleaned.csv'  # Update the path if needed
data = pd.read_csv(file_path)

# Inspect the data (optional)
print(data.head())

# Selecting the features
features = ["Flower Name", "MRP (₹)", "Customer Segment"]

# Encode the categorical variables ("Flower Name" and "Customer Type")
label_encoders = {}
for col in ["Flower Name", "Customer Segment"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare the data for clustering
X = data[features]

# Standardize the data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust hyperparameters as needed
labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = labels

# Evaluate clustering with Silhouette Score (only valid for more than one cluster)
if len(set(labels)) > 1 and -1 not in set(labels):  # Avoid invalid silhouette score with noise or single cluster
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("Silhouette Score not applicable (single cluster or too much noise).")

# Plot clusters for visualization (use first two features for simplicity)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_scaled[:, 0], y=X_scaled[:, 1],
    hue=labels, palette="viridis", legend="full"
)
plt.title("DBSCAN Clustering Visualization")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend(title="Cluster")

# Save the plot as an image
plt.savefig('dbscan_clusters.png', dpi=300, bbox_inches='tight')
print("Cluster visualization saved as 'dbscan_clusters.png'.")


# Save the clustered data
data.to_csv('flowers_clustered.csv', index=False)
