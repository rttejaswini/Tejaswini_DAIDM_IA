import pandas as pd
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.graph_objects as go
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('assets/charts', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Load data
df = pd.read_csv('data/raw/ai_adoption_clients_250.csv')  # Copy uploaded file here
date_cols = ['login_date', 'onboarding_date', 'last_ai_usage_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')

today = pd.to_datetime('2026-03-13')
df['Recency'] = (today - df['last_ai_usage_date']).dt.days
df['Frequency'] = df['ai_feature_usage_hours'] / ((today - df['onboarding_date']).dt.days / 7)
df['Monetary'] = df['contract_value'] * (df['ai_feature_usage_hours'] / 100)

# Scale RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']].fillna(0))

# Train K-Means (5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(rfm_scaled)

# Train Hierarchical
hc = AgglomerativeClustering(n_clusters=5)
df['hc_cluster'] = hc.fit_predict(rfm_scaled)

# Save best model bundle
model_bundle = {'kmeans': kmeans, 'scaler': scaler, 'hc': hc}
joblib.dump(model_bundle, 'models/best_model.pkl')

# Association Rules
df_disc = pd.get_dummies(df[['industry', 'company_size', 'subscription_tier', 'churn_status']])
freq = apriori(df_disc, min_support=0.1, use_colnames=True)
rules = association_rules(freq, 'confidence', 0.6)
rules.to_csv('assets/charts/rules.csv', index=False)

print("✅ Models trained & saved!")
print("📊 Cluster distribution:", df['cluster'].value_counts().to_dict())
print(f"🎯 Silhouette: {silhouette_score(rfm_scaled, df['cluster']):.3f}")
