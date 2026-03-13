import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import joblib

# Page config
st.set_page_config(page_title="AI Adoption Analytics", layout="wide")
st.title("🚀 AI Adoption Analytics Platform")
st.markdown("**Descriptive → Diagnostic → Predictive → Prescriptive Analytics**")

# Data Loading with Error Handling
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    date_cols = ['login_date', 'onboarding_date', 'last_ai_usage_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
    return df

# Sidebar Data Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success(f"✅ Loaded {len(df)} clients")
else:
    try:
        df = load_data('ai_adoption_clients_250.csv')
        st.sidebar.info("Using sample data (250 clients)")
    except Exception as e:
        st.error("⚠️ Please upload a CSV file to begin.")
        st.stop()

# RFM Computation
today = pd.to_datetime('2026-03-13')
df['Recency'] = (today - df['last_ai_usage_date']).dt.days.clip(0)
df['Frequency'] = df['ai_feature_usage_hours'] / ((today - df['onboarding_date']).dt.days / 7).clip(1)
df['Monetary'] = df['contract_value'] * (df['ai_feature_usage_hours'] / 100)

# Fill NA values safely for clustering
cluster_features = df[['Recency', 'Frequency', 'Monetary']].fillna(0)

# ==========================================
# 🎯 DIAGNOSTIC ANALYTICS: CLUSTERING
# ==========================================
st.header("🎯 Diagnostic Analytics: Clustering")

# Compact KPI Row
col1, col2, col3, col4 = st.columns(4)
n_clusters = col1.slider("K-Means Clusters", 3, 7, 5)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(cluster_features)

# K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)
sil_score = silhouette_score(rfm_scaled, df['Cluster'])

col2.metric("Silhouette Score", f"{sil_score:.3f}")
col3.metric("AI Adoption Rate", f"{(df['ai_feature_usage_hours'] > 20).mean():.1%}")
col4.metric("Avg Contract Value", f"${df['contract_value'].mean():,.0f}")

# Layout for Charts (Side-by-Side)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # 3D Scatter
    fig_3d = px.scatter_3d(
        df, x='Recency', y='Frequency', z='Monetary',
        color='Cluster', hover_data=['client_id', 'industry'],
        title="RFM Analysis (Interactive 3D)", opacity=0.7
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)

with chart_col2:
    # Cluster Distribution Pie Chart
    cluster_pct = df['Cluster'].value_counts(normalize=True).sort_index() * 100
    fig_pie = px.pie(
        values=cluster_pct.values, 
        names=[f"Cluster {i}" for i in cluster_pct.index], 
        title="Client Segment Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# 🌳 HIERARCHICAL CLUSTERING (FIXED DENDROGRAM)
# ==========================================
st.subheader("🌳 Hierarchical Clustering (Ward Linkage)")

# Manual Dendrogram creation using Scipy -> Plotly (Much cleaner than figure_factory)
Z = linkage(rfm_scaled, 'ward')
# Create a dummy dendrogram just to extract coordinates
dendro = dendrogram(Z, no_plot=True)

icoord = np.array(dendro['icoord'])
dcoord = np.array(dendro['dcoord'])

fig_dendro = go.Figure()
for i, d in zip(icoord, dcoord):
    fig_dendro.add_trace(go.Scatter(
        x=i, y=d, mode='lines', line=dict(color='blue'), showlegend=False
    ))

fig_dendro.update_layout(
    height=400, title="Dendrogram",
    xaxis=dict(showticklabels=False, title="Clients"),
    yaxis=dict(title="Distance (Ward)")
)
st.plotly_chart(fig_dendro, use_container_width=True)


# ==========================================
# 🔄 CLIENT JOURNEY & PATTERNS
# ==========================================
pattern_col1, pattern_col2 = st.columns(2)

with pattern_col1:
    st.subheader("🔗 Client Pattern Insights")
    # Safe Correlation Table (No external libraries needed)
    corr_data = pd.crosstab(df['industry'], df['churn_status'], normalize='index').round(3)
    corr_data['Avg_AI_Hours'] = df.groupby('industry')['ai_feature_usage_hours'].mean().round(1)
    st.dataframe(corr_data.style.background_gradient(cmap='Blues').format({
        'Active': '{:.1%}', 'At-Risk': '{:.1%}', 'Churned': '{:.1%}', 'Avg_AI_Hours': '{:.1f}'
    }))

with pattern_col2:
    st.subheader("📊 Industry vs Size Heatmap")
    fig_heatmap = px.density_heatmap(
        df, x='industry', y='company_size', z='ai_feature_usage_hours', 
        histfunc="avg", color_continuous_scale='Viridis',
        title="Avg AI Usage: Industry vs Size"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ==========================================
# 👥 PRESCRIPTIVE: PERSONAS & SIMULATOR
# ==========================================
st.header("👥 Client Personas & Prescriptive Actions")
selected_cluster = st.selectbox("Select Cluster to View Strategy:", sorted(df['Cluster'].unique()))
cluster_df = df[df['Cluster'] == selected_cluster]

col_p1, col_p2 = st.columns(2)
with col_p1:
    personas = [
        "🏆 Champions: High Value, Highly Engaged",
        "⚡ Experimenters: High Potential, Needs Nurturing", 
        "😴 Laggards: Low Usage, High Risk",
        "💀 Lost/At-Risk: Immediate Intervention Required",
        "🚀 Leaders: Top Tier Adopters"
    ]
    st.markdown(f"### {personas[selected_cluster % 5]}")
    st.metric("Clients in Segment", len(cluster_df))
    st.metric("Segment Avg AI Hours", f"{cluster_df['ai_feature_usage_hours'].mean():.1f}")

with col2:
    st.markdown("### 📋 Recommended Playbook")
    st.markdown("""
    1. **Immediate Outreach:** Schedule personalized AI onboarding.
    2. **Incentive:** Offer 15% discount on next renewal for hitting usage milestones.
    3. **Education:** Send industry-specific case studies.
    """)

st.header("🔮 Revenue Simulator")
discount = st.slider("Targeted Discount %", 0, 30, 10)
budget = st.slider("Workshop Budget ($K)", 0, 100, 25)

# Simple uplift math based on inputs
uplift_pct = (discount * 0.5 + budget * 0.2) / 100
total_val = df['contract_value'].sum()
st.success(f"**Projected Revenue Uplift:** +{uplift_pct:.1%} (Est. +${(total_val * uplift_pct):,.0f})")

# ==========================================
# 📤 EXPORT
# ==========================================
st.markdown("---")
if st.button("📥 Download Prioritized Outreach CSV"):
    outreach = df[df['churn_status'].isin(['At-Risk', 'Churned'])].head(50)
    outreach = outreach[['client_id', 'industry', 'churn_status', 'ai_feature_usage_hours', 'contract_value']]
    outreach['Recommended_Action'] = 'Offer AI Workshop + 15% Discount'
    
    csv = outreach.to_csv(index=False)
    st.download_button(
        label="Click to Download",
        data=csv,
        file_name="ai_adoption_outreach_targets.csv",
        mime="text/csv",
    )
