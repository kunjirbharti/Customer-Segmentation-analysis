# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Create a sample dataset
np.random.seed(42)
n_customers = 200

data = {
    'CustomerID': np.arange(1, n_customers + 1),
    'Age': np.random.randint(18, 70, size=n_customers),
    'Gender': np.random.choice(['Male', 'Female'], size=n_customers),
    'Annual Income (k$)': np.random.randint(15, 150, size=n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, size=n_customers),
    'Tenure (years)': np.random.randint(1, 10, size=n_customers)
}

df = pd.DataFrame(data)

# Step 3: Data preprocessing
# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Select features for clustering
X = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)', 'Tenure (years)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find optimal number of clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method - Finding Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Step 5: Apply KMeans with 5 clusters (example)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster info to dataframe
df['Cluster'] = y_kmeans

# Step 6: Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_kmeans, palette='Set2', s=70)
plt.title('Customer Segments Visualization (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Step 7: Cluster description
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# Optional: Export dataset with clusters
df.to_csv('customer_segments.csv', index=False)
print("Dataset with clusters saved as 'customer_segments.csv'")
