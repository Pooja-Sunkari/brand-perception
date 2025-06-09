#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

# 🪄 Title of our magical app
st.title("PerceptoMap")
st.write("Welcome to PerceptoMap! Upload your customer perception data, and see a magic 2D map that shows how brands are similar and which groups they belong to.")

# 📂 Upload section for the data file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file as a DataFrame
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # 🎨 Optional filtering if columns exist
    # Filter by customer segment if the column "segment" exists.
    if 'segment' in data.columns:
        segments = data['segment'].unique()
        selected_segment = st.selectbox("Select Customer Segment", segments)
        data = data[data['segment'] == selected_segment]

    # Filter by region if the column "region" exists.
    if 'region' in data.columns:
        regions = data['region'].unique()
        selected_region = st.selectbox("Select Region", regions)
        data = data[data['region'] == selected_region]

    # Filter by time if the column "time" exists.
    if 'time' in data.columns:
        times = data['time'].unique()
        selected_time = st.selectbox("Select Time", times)
        data = data[data['time'] == selected_time]

    # 📏 Prepare the data for MDS
    # We assume that your CSV contains similarity data as numeric columns.
    # If you have a special "brand" column (names), we keep it aside.
    if 'brand' in data.columns:
        brands = data['brand']
    else:
        # If no brand names exist, we create generic ones.
        brands = pd.Series(["Brand " + str(i) for i in range(len(data))])
    
    # Select numeric columns only (assumed to be perception ratings)
    numeric_data = data.select_dtypes(include=[np.number])
    # Note: if your file contains a precomputed distance matrix instead of individual features,
    # you might need to adjust this part (or convert similarity into distance).

    # 🧠 Apply Multidimensional Scaling (MDS) to map our brands on a 2D canvas.
    mds = MDS(n_components=2, random_state=42)
    mds_coords = mds.fit_transform(numeric_data)

    # Create a DataFrame to hold the 2D coordinates along with brand names.
    plot_data = pd.DataFrame(mds_coords, columns=["Dim1", "Dim2"])
    plot_data["brand"] = brands

    # 👯‍♀️ Use K-Means clustering to see which brands are best friends!
    k = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(mds_coords)
    plot_data['cluster'] = clusters

    # 📊 Draw a scatter plot that shows the brands on our magical map.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="Dim1", y="Dim2", hue="cluster", data=plot_data,
                    palette="Set1", style="cluster", s=100, ax=ax)
    
    # Mark the cluster centers (like where the friends hang out together!)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Center')
    ax.set_title("2D Perceptual Map with Clusters")
    ax.legend()
    st.pyplot(fig)

    # 🕵️‍♀️ Highlighting market gaps:
    # One simple idea: look for clusters that are very small (few brands) as they might be 'empty' or underpopulated.
    st.header("Repositioning Suggestions")
    suggestions = []
    for i, group in plot_data.groupby('cluster'):
        if len(group) < 3:
            # This cluster has very few brands: suggest they consider repositioning.
            brand_list = ", ".join(group["brand"].tolist())
            suggestions.append(f"* Cluster {i} is small (brands: {brand_list}). Consider strategies to strengthen or reposition this segment.")
    
    if suggestions:
        for suggestion in suggestions:
            st.markdown(suggestion)
    else:
        st.write("All clusters look balanced. No repositioning suggestions at this time.")

    # 🎉 End of the app actions
    st.write("Enjoy exploring your PerceptoMap!")


# In[ ]:




