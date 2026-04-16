# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==============================
# 2. TITLE
# ==============================

st.title("🛒 Customer Segmentation using K-Means")
st.write("Customer data is loaded automatically")


# ==============================
# 3. LOAD DATASET (NO UPLOAD)
# ==============================

data = pd.read_csv("Mall_Customers.csv")

st.subheader("📊 Dataset Preview")
st.write(data.head())


# ==============================
# 4. HANDLE MISSING VALUES
# ==============================

st.subheader("🔍 Missing Values Check")
st.write(data.isnull().sum())

# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)
data.dropna(inplace=True)

st.write("✅ Missing values handled!")


# ==============================
# 5. PREPROCESSING
# ==============================

# Convert Gender column
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)

# Select features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

if 'Gender' in data.columns:
    features.insert(0, 'Gender')

X = data[features]


# ==============================
# 6. SCALING
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# 7. SELECT K
# ==============================

k = st.slider("Select number of clusters (K)", 2, 10, 5)


# ==============================
# 8. APPLY K-MEANS
# ==============================

kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans


# ==============================
# 9. SHOW RESULT
# ==============================

st.subheader("📌 Clustered Data")
st.write(data.head())


# ==============================
# 10. VISUALIZATION
# ==============================

st.subheader("📈 Cluster Visualization")

plt.figure()
plt.scatter(data['Annual Income (k$)'],
            data['Spending Score (1-100)'],
            c=y_kmeans)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")

st.pyplot(plt)


# ==============================
# 11. DOWNLOAD
# ==============================

csv = data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Clustered Data",
    data=csv,
    file_name="clustered_customers.csv",
    mime='text/csv'
)