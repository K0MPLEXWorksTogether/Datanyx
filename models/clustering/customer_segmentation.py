# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (your given dataset)
data = {
    "Flower": ['Marigold', 'Rose', 'Jasmine', 'Lily', 'Sunflower', 'Tulip', 'Orchid', 'Gerbera', 
               'Dahlia', 'Chrysanthemum', 'Lavender', 'Carnation', 'Hibiscus', 'Poppy', 'Geranium', 
               'Zinnia', 'Daisy', 'Calla Lily'],
    "Price": [150, 120, 100, 180, 130, 250, 300, 110, 170, 140, 200, 160, 90, 220, 140, 100, 110, 250],
    "Quantity": [200, 150, 100, 80, 120, 50, 30, 180, 70, 90, 60, 130, 160, 40, 110, 130, 140, 60],
    "Weather": ['Sunny', 'Sunny', 'Windy', 'Sunny', 'Sunny', 'Windy', 'Windy', 'Sunny', 'Sunny', 'Windy', 
                'Windy', 'Sunny', 'Sunny', 'Windy', 'Sunny', 'Sunny', 'Sunny', 'Windy'],
    "Events": [
        'Marriage, Temple', 'Daily Consumers, Weddings', 'Temple, Marriage', 'Wedding, Special Events', 
        'Wedding, Festivals', 'High-End Customers, Special Occasions', 'High-End Customers, Luxury Events', 
        'Weddings, Daily Consumers', 'Special Events, Daily Consumers', 'Temple, Festivals', 
        'Luxury Events, Special Occasions', 'Weddings, Corporate Events', 'Temple, Daily Consumers', 
        'Luxury Events, Special Occasions', 'Weddings, Corporate Events', 'Daily Consumers, Festivals', 
        'Weddings, Daily Consumers', 'Weddings, Daily Consumers'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# One-hot encode 'Weather' and 'Events' columns
weather_encoded = pd.get_dummies(df['Weather'], drop_first=True)
events_encoded = pd.get_dummies(df['Events'], drop_first=True)

# Concatenate all the data
df_encoded = pd.concat([df[['Price', 'Quantity']], weather_encoded, events_encoded], axis=1)

# Normalize the numerical columns (Price, Quantity)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded[['Price', 'Quantity']])

# Concatenate scaled data with encoded columns
final_data = np.concatenate([scaled_data, df_encoded.drop(['Price', 'Quantity'], axis=1).values], axis=1)

# Now we have a processed dataset ready for clustering


# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters (3 here)
kmeans_labels = kmeans.fit_predict(final_data)

# Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(final_data, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette}")

# Add KMeans labels to the DataFrame
df['KMeans_Cluster'] = kmeans_labels

# Visualize K-Means clustering with PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_components = pca.fit_transform(final_data)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('K-Means.png', dpi=300, bbox_inches='tight')

# Hierarchical Clustering (Agglomerative)
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(final_data)

# Silhouette Score for Hierarchical Clustering
hierarchical_silhouette = silhouette_score(final_data, hierarchical_labels)
print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette}")

# Add Hierarchical labels to the DataFrame
df['Hierarchical_Cluster'] = hierarchical_labels

# Visualize Hierarchical Clustering
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=hierarchical_labels, cmap='plasma')
plt.title('Hierarchical Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("Hierarchical.png", dpi=300, bbox_inches='tight')

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(final_data)

# Silhouette Score for DBSCAN
# DBSCAN assigns -1 to noise points, so we exclude those from the score calculation.
dbscan_silhouette = silhouette_score(final_data, dbscan_labels[dbscan_labels != -1])
print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")

# Add DBSCAN labels to the DataFrame
df['DBSCAN_Cluster'] = dbscan_labels

# Visualize DBSCAN Clustering
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan_labels, cmap='coolwarm')
plt.title('DBSCAN Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("DBSCAN.png", dpi=300, bbox_inches='tight')

# Optionally, print the final DataFrame with cluster labels
print(df)
