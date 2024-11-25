import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from PIL import Image

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

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters (3 here)
kmeans_labels = kmeans.fit_predict(final_data)

# Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(final_data, kmeans_labels)

# Add KMeans labels to the DataFrame
df['KMeans_Cluster'] = kmeans_labels

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(final_data)

# Plot K-Means Clustering
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('K-Means.png', dpi=300, bbox_inches='tight')
plt.close()

# Streamlit app
st.title("Flower Clustering - K-Means Only")

# Display Silhouette Score
st.subheader(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")

# Display the clustering results as a table
st.subheader("Clustering Results")
st.dataframe(df)

# Display the clustering visualization
st.subheader("K-Means Clustering Visualization")
kmeans_image = Image.open('K-Means.png')
st.image(kmeans_image, caption='K-Means Clustering')
