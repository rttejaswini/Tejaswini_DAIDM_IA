import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from datetime import datetime
import os

st.set_page_config(page_title="AI Adoption Analytics", layout="wide")
st.title("🚀 AI Adoption Analytics Platform")
st.markdown("**K-Means • Hierarchical Clustering • RFM Analysis • Client Personas**")

# Data loading (robust)
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    date_cols = ['login_date', 'onboarding_date', 'last_ai_usage_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
    return df.fillna(0)

# Sidebar
st.sidebar.header("📁 Data")
if st.sidebar.file_uploader("Upload CSV", type="csv") is not None:
    df = load_data(st.sidebar.file_uploader("Upload CSV", type="csv"))
else:
    try:
        df = load_data('ai_adoption_clients_250.csv')
        st.sidebar.success(f"✅ Loaded {len(df)} clients")
    except:
        st.error("Place ai_adoption_clients_250.csv in root")
        st.stop()

today = pd.to_datetime('2026-03-13')

# RFM Computation
df['Recency'] = (today - df['last_ai_usage_date']).dt.days.clip(0)
df['Frequency'] = df['ai_feature_usage_hours'] / ((today - df['onboarding_date']).dt.days / 7).clip(1)
df['Monetary'] = df['contract_value'] * df['ai_feature_usage_hours'] / 100

# Clustering Section (COMPACT)
st.header("🎯 Clustering Analysis")
col1, col2, col3 = st.columns(3)
n_clusters = col1.slider("Clusters", 3, 6, 4, key="nclust")

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency','Frequency','Monetary']])

kmeans = KMeans(n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)
sil_score = silhouette_score(rfm_scaled, df['Cluster'])

col2.metric("Silhouette", f"{sil_score:.3f}")
col3.metric("Adoption Rate", f"{(df['ai_feature_usage_hours']>15).mean():.0%}")

# Compact Cluster Distribution
st.subheader("📊 Cluster Distribution")
cluster_pct = df['Cluster'].value_counts(normalize=True).sort_index() * 100
fig_pie = px.pie(values=cluster_pct.values, names=[f"Cluster {i}" for i in cluster_pct.index], 
                 title="Client Segments")
st.plotly_chart(fig_pie, use_container_width=True)

# RFM 3D (IMPROVED)
fig_3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary', color='Cluster',
                      hover_data=['client_id','industry','churn_status'],
                      title="RFM Analysis (Interactive 3D)", opacity=0.7)
fig_3d.update_traces(marker_size=4)
st.plotly_chart(fig_3d, use_container_width=True)

# FIXED Dendrogram (LEGIBLE + INTERACTIVE)
st.subheader("🌳 Hierarchical Clustering")
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(rfm_scaled, 'ward')
fig_dendro = go.Figure(data=go.Scatter(x=Z[:,0], y=Z[:,1], mode='lines'))
fig_dendro.update_layout(height=500, title="Hierarchical Dendrogram (Ward)", 
                        xaxis_title="Distance", yaxis_title="Height")
st.plotly_chart(fig_dendro, use_container_width=True)

# FIXED Sankey (Safe Columns)
st.subheader("🔄 Client Journey")
if all(col in df.columns for col in ['industry', 'subscription_tier', 'churn_status']):
    fig_sankey = px.sankey(df, path=['industry', 'subscription_tier', 'churn_status'],
                          title="Journey Flow")
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("Sankey: Add industry/tier/churn columns")

# Heatmap (COMPACT)
fig_heatmap = px.imshow(df.pivot_table(values='ai_feature_usage_hours', 
                                      index='industry', columns='company_size', aggfunc='mean'),
                       title="AI Usage Heatmap", aspect="auto", color_continuous_scale='RdYlGn')
st.plotly_chart(fig_heatmap, use_container_width=True)

# Personas (IMPROVED)
st.header("👥 Client Personas")
cluster_sel = st.selectbox("Select Cluster", sorted(df['Cluster'].unique()))
cluster_data = df[df['Cluster'] == cluster_sel]

col1, col2 = st.columns([2,1])
with col1:
    st.metric("Size", len(cluster_data), delta=f"{len(cluster_data)/len(df)*100:.0f}%")
    st.metric("Avg AI Hours", f"{cluster_data['ai_feature_usage_hours'].mean():.1f}")
    st.metric("Churn Risk", f"{(cluster_data['churn_status']=='Churned').mean():.0%}")
with col2:
    personas = ["🏆 Champions", "⚡ Experimenters", "😴 Laggards", "💀 Lost", "🚀 Leaders"]
    st.markdown(f"### {personas[cluster_sel % 5]}")
    st.markdown("""
    • **Workshops + Demos**
    • **Tier Upgrade Offers**  
    • **Success Case Studies**
    """)

# What-If (ENHANCED)
st.header("🎯 Revenue Simulator")
col1, col2 = st.columns(2)
discount = col1.slider("💰 Discount %", 0, 25, 10)
workshops = col2.slider("🎓 Workshops ($K)", 0, 50, 20)

total_contracts = df['contract_value'].sum() / 1e6  # $M
uplift = (discount*0.8 + workshops*0.4) / 100
st.metric("Revenue Uplift", f"+{uplift:.1%}", delta=f"+${uplift*total_contracts:.1f}M")

# Export
if st.button("📥 Download Outreach List (Top 50 At-Risk)"):
    risky = df[df['churn_status'].isin(['At-Risk','Churned'])].head(50)
    risky['Action'] = 'AI Workshop + 15% Discount'
    csv = risky[['client_id','industry','ai_feature_usage_hours','contract_value','Action']].to_csv(index=False)
    st.download_button("Download CSV", csv, "outreach.csv", "text/csv")

st.markdown("---")
st.caption("🎓 Data Mining Course: RFM • K-Means • Hierarchical Clustering")
