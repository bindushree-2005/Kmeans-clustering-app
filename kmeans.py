# ==============================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================

import pandas as pd                      # For handling dataset (CSV file)
import matplotlib.pyplot as plt          # For plotting graphs
from sklearn.cluster import KMeans       # K-Means clustering algorithm
from sklearn.preprocessing import StandardScaler  # For scaling data


# ==============================
# 2. LOAD DATASET
# ==============================

# Load the dataset (make sure file is in same folder)
data = pd.read_csv("Mall_Customers.csv")

# Display first 5 rows to understand data
print("First 5 rows of dataset:\n")
print(data.head())

# Show column names
print("\nColumn names:\n", data.columns)


# ==============================
# 3. DATA PREPROCESSING
# ==============================

# Convert Gender column into numeric values
# Male = 1, Female = 0
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Check if conversion worked
print("\nAfter converting Gender:\n")
print(data.head())


# ==============================
# 4. SELECT FEATURES FOR CLUSTERING
# ==============================

# We select only useful columns (ignore CustomerID)
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Print selected data
print("\nSelected Features:\n")
print(X.head())


# ==============================
# 5. FEATURE SCALING
# ==============================

# Scaling ensures all features are on same scale
# (important for distance-based algorithms like K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data Sample:\n", X_scaled[:5])


# ==============================
# 6. FIND OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# ==============================

wcss = []  # List to store WCSS values

# Try cluster values from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)   # Inertia = WCSS

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# From graph, choose optimal K (usually 4 or 5)


# ==============================
# 7. APPLY K-MEANS ALGORITHM
# ==============================

# Create model with chosen K (here we use 5)
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit model and assign cluster to each data point
y_kmeans = kmeans.fit_predict(X_scaled)


# ==============================
# 8. ADD CLUSTER RESULTS TO DATASET
# ==============================

# Add new column 'Cluster'
data['Cluster'] = y_kmeans

print("\nDataset with Cluster Column:\n")
print(data.head())
data.to_csv("clustered_customers.csv", index=False)

# ==============================
# 9. VISUALIZE CLUSTERS (2D GRAPH)
# ==============================

# Plot clusters using Income vs Spending Score
plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=y_kmeans)   # Color based on cluster

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# ==============================
# 10. SAVE OUTPUT DATA (OPTIONAL)
# ==============================

# Save new dataset with clusters
data.to_csv("clustered_customers.csv", index=False)

print("\nClustered data saved as 'clustered_customers.csv'")