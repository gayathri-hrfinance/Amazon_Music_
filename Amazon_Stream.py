import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Load model and data ---
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
df = pd.read_csv('single_genre_artists.csv')

# --- Define features used for clustering ---
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min']

# --- App title ---
st.title("🎵 Amazon Music Song Cluster Predictor")

st.markdown("Enter audio features of a song to predict its cluster and visualize it in PCA space.")

# --- Feature sliders based on real data ranges ---
inputs = {}
for f in features:
    min_v = float(df[f].min())
    max_v = float(df[f].max())
    default_v = float(df[f].median())
    step_v = 0.01 if f in ['danceability', 'energy', 'speechiness', 'acousticness',
                           'instrumentalness', 'liveness', 'valence'] else 0.1 if f == 'loudness' else 1.0 if f == 'tempo' else 0.1
    inputs[f] = st.slider(f, min_value=min_v, max_value=max_v, value=default_v, step=step_v)

# --- Predict cluster ---
input_df = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_df)
cluster = kmeans.predict(input_scaled)[0]

# --- Display prediction ---
st.subheader(f"🎧 Predicted Cluster: {cluster}")
if cluster == 0:
    st.write("🟢 Calm Acoustic Songs")
elif cluster == 1:
    st.write("🟡 Energetic Pop/Dance Tracks")
else:
    st.write("🔵 Spoken Word / Rap-like Songs")

# --- PCA Visualization ---
df_clean = df[features].dropna()
X_scaled = scaler.transform(df_clean)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
cluster_labels = kmeans.predict(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.25)
input_pca = pca.transform(input_scaled)
ax.scatter(input_pca[:, 0], input_pca[:, 1], c='red', s=150, edgecolor='black', label='Your Song')

ax.set_title("🎨 PCA Visualization of Song Clusters")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend()

st.pyplot(fig)
