#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

# 🪄 Title of our magical app
st.set_page_config(page_title="PerceptoMap", layout="wide")
st.title("🗺️ PerceptoMap")
st.write("Welcome to PerceptoMap! Upload customer perception data and see a fun 2D map that shows how brands are grouped in the market.")

# 📂 Upload section
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

    # 🎛️ Optional filters
    if 'segment' in data.columns:
        selected_segment = st.selectbox("👥 Select Customer Segment", ["All"] + sorted(data['segment'].dropna().unique().tolist()))
        if selected_segment != "All":
            data = data[data['segment'] == selected_segment]

    if 'region' in data.columns:
        selected_region = st.selectbox("🌎 Select Region", ["All"] + sorted(data['region'].dropna().unique().tolist()))
        if selected_region != "All":
            data = data[data['region'] == selected_region]

    if 'time' in data.columns:
        selected_time = st.selectbox("🕒 Select Time", ["All"] + sorted(data['time'].dropna().unique().tolist()))
        if selected_time != "All":
            data = data[data['time'] == selected_time]

    # 🏷️ Get brand names
    if 'brand' in data.columns:
        brands = data['brand']
    else:
        brands = pd.Series(["Brand " + str(i) for i in range(len(data))])

    # 🔢 Only numeric columns (the perception features)
    features = data.select_dtypes(include=[np.number])
    
    # 🧠 MDS - reduce to 2D
    mds = MDS(n_components=2, random_state=42)
    coords = mds.fit_transform(features)
    plot_data = pd.DataFrame(coords, columns=["Dim1", "Dim2"])
    plot_data["brand"] = brands.values

    # 🎯 Clustering
    k = st.slider("🎨 How many clusters?", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    plot_data["cluster"] = kmeans.fit_predict(coords)

    # 🖼️ Clear & Fun Map
    st.subheader("🖼️ Brand Perception Map (Easy View)")

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_colors = plt.cm.get_cmap('tab10', k)

    for i, row in plot_data.iterrows():
        ax.scatter(row["Dim1"], row["Dim2"], 
                   s=300, 
                   color=cluster_colors(row["cluster"]), 
                   alpha=0.85, edgecolors="black")
        ax.text(row["Dim1"], row["Dim2"], row["brand"], 
                ha='center', va='center', fontsize=9, color='black', weight='bold')

    ax.set_title("🎈 Where Each Brand Stands in the Market", fontsize=16)
    ax.set_xlabel("Dimension 1 (🧠 Style + Innovation)", fontsize=12)
    ax.set_ylabel("Dimension 2 (❤️ Comfort + Popularity)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 🌈 Legend
    for cluster_id in range(k):
        ax.scatter([], [], color=cluster_colors(cluster_id), label=f"Group {cluster_id+1}")
    ax.legend(title="Friend Circles")

    st.pyplot(fig)

    # 💡 Repositioning Suggestions
    st.subheader("💡 Repositioning Suggestions")
    suggestions = []
    for i, group in plot_data.groupby('cluster'):
        if len(group) < 3:
            brand_list = ", ".join(group["brand"])
            suggestions.append(f"🚨 Group {i+1} is small (brands: {brand_list}). Try new ideas or move closer to other groups!")

    if suggestions:
        for s in suggestions:
            st.markdown(s)
    else:
        st.write("✅ All groups look balanced! No repositioning needed.")

    # 🎉 The end
    st.markdown("---")
    st.write("🎉 Done exploring! Try uploading different data to discover new insights.")


# In[ ]:




