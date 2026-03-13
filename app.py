import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import base64
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="AI Adoption Analytics", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 AI Adoption Analytics Platform")
st.markdown("**Descriptive → Diagnostic → Predictive → Prescriptive Analytics for B2B SaaS Clients**")

# Load pre-trained model (fallback to compute live)
@st.cache_data
def load_model():
    try:
        return joblib.load('models/best_model.pkl')
    except:
        return None

model = load_model()

# Sidebar
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Fix date formats (DD-MM-YYYY)
        date_cols = ['login_date', 'onboarding_date', 'last_ai_usage_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        return df
    
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.sidebar.success(f"Loaded {len(df)} clients")
else:
    # Use attached file
    try:
        df = pd.read_csv('data/raw/ai_adoption_clients_250.csv')
        date_cols = ['login_date', 'onboarding_date', 'last_ai_usage_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        st.session_state.df = df
        st.sidebar.info("Using sample data (250 clients)")
    except:
        st.error("Upload CSV or place data/raw/ai_adoption_clients_250.csv")
        st.stop()

df = st.session_state.df.copy()
today = pd.to_datetime('2026-03-13')

# Compute RFM
def compute_rfm(df, today):
    df['Recency'] = (today - df['last_ai_usage_date']).dt.days
    df['Frequency'] = df['ai_feature_usage_hours'] / ((today - df['onboarding_date']).dt.days / 7)  # weekly
    df['Monetary'] = df['contract_value'] * (df['ai_feature_usage_hours'] / 100)
    return df

df = compute_rfm(df, today)

# Clustering
st.header("🎯 Diagnostic Analytics: Clustering")
col1, col2, col3 = st.columns(3)
n_clusters = col1.slider("K-Means Clusters", 3, 7, 5)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']].fillna(0))

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

hc = AgglomerativeClustering(n_clusters=n_clusters)
df['HC_Cluster'] = hc.fit_predict(rfm_scaled)

sil_score = silhouette_score(rfm_scaled, df['KMeans_Cluster'])
col2.metric("Silhouette Score", f"{sil_score:.3f}")
col3.metric("AI Adoption Rate", f"{(df['ai_feature_usage_hours']>20).mean():.1%}")

# KPIs Row
col1, col2, col3, col4 = st.columns(4)
cluster_dist = df['KMeans_Cluster'].value_counts(normalize=True).round(2)
for i, (cluster, pct) in enumerate(cluster_dist.head(4).items()):
    col1.metric(f"Cluster {cluster}", f"{pct:.0%}")

# RFM 3D Scatter
fig_3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary',
                       color='KMeans_Cluster', hover_data=['client_id', 'industry'],
                       title="RFM 3D Scatter (K-Means Clusters)")
fig_3d.update_traces(marker=dict(size=5))
st.plotly_chart(fig_3d, use_container_width=True)

# Dendrogram
fig_dendro = ff.create_dendrogram(rfm_scaled, color_threshold=1.5, orientation='right')
fig_dendro.update_layout(height=600, title="Hierarchical Clustering Dendrogram")
st.plotly_chart(fig_dendro, use_container_width=True)

# Association Rules
st.subheader("🔗 Association Rules (Apriori)")
# Discretize for Apriori
df_disc = pd.get_dummies(df[['industry', 'company_size', 'subscription_tier', 'churn_status']])
freq_items = apriori(df_disc, min_support=0.1, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
rules = rules.nlargest(20, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
st.dataframe(rules.style.format({'support': '{:.2%}', 'confidence': '{:.2%}', 'lift': '{:.2f}'}).background_gradient())

# Sankey
st.subheader("📊 Client Journey Sankey")
fig_sankey = px.sankey(df, path=['industry', 'subscription_tier', 'churn_status'],
                       title="Industry → Tier → Churn")
st.plotly_chart(fig_sankey)

# Heatmap
fig_heatmap = px.density_heatmap(df, x='industry', y='company_size', 
                                z='ai_feature_usage_hours', color_continuous_scale='Viridis',
                                title="Industry vs Size: AI Usage Heatmap")
st.plotly_chart(fig_heatmap)

# Client Explorer
st.header("👥 Client Segments & Personas")
selected_cluster = st.selectbox("Select Cluster", sorted(df['KMeans_Cluster'].unique()))
cluster_df = df[df['KMeans_Cluster'] == selected_cluster]

st.subheader(f"Cluster {selected_cluster} Profile")
col1, col2 = st.columns(2)
with col1:
    persona = {
        0: "🏆 AI Champions: High usage, low support needs",
        1: "⚡ Experimenters: Growing adoption, needs nurturing", 
        2: "😴 Laggards: Low engagement, high churn risk",
        3: "💀 Lost: Minimal usage, already churned",
        4: "🚀 Enterprise Leaders: High value, full adoption"
    }.get(selected_cluster % 5, "Emerging Adopters")
    st.markdown(f"### {persona}")
    st.metric("Clients", len(cluster_df))
    avg_ai = cluster_df['ai_feature_usage_hours'].mean()
    st.metric("Avg AI Hours", f"{avg_ai:.1f}")
with col2:
    playbook = [
        "Personalized AI workshops",
        "Feature onboarding sessions", 
        "Tier upgrade incentives",
        "Success story case studies"
    ]
    for action in playbook[:3]:
        st.markdown(f"• **{action}**")

# What-If Simulator
st.header("🔮 What-If Simulator")
discount = st.slider("Discount %", 0, 30, 10)
workshop_budget = st.slider("Workshop Investment ($K)", 0, 100, 25)
uplift = (discount / 10 + workshop_budget / 50) * 0.15  # Simple model
st.metric("Projected Revenue Uplift", f"+{uplift:.1%}", delta=f"+${uplift*df['contract_value'].sum()/1000:.0f}K")

# Export
st.header("📤 Export Outreach List")
if st.button("Generate Outreach CSV"):
    outreach = df[df['churn_status'].isin(['At-Risk', 'Churned'])].head(50)[
        ['client_id', 'industry', 'churn_status', 'ai_feature_usage_hours', 'contract_value']]
    outreach['Recommended Offer'] = 'AI Workshop + 15% Discount'
    csv = outreach.to_csv(index=False)
    st.download_button("Download CSV", csv, "ai_adoption_outreach.csv", "text/csv")

st.markdown("---")
st.caption("🎓 Built for Data Mining Course: K-Means, Hierarchical Clustering, Association Rules")
