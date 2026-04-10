import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px

def run_clustering_analysis(df: pd.DataFrame, n_clusters: int = 3):
    """
    Performs K-Means clustering on the dataset and reduces dimensions for visualization.
    """
    # 1. Prepare Data (Only numeric for clustering)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for clustering.")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(numeric_df)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 2. Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 3. PCA for Visualization (Reduce to 2D)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    # 4. Create Result DataFrame
    plot_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    plot_df['Cluster'] = [f"Segment {c}" for c in clusters]
    
    # Add some original columns for tooltips
    for col in numeric_df.columns[:3]:
        plot_df[col] = numeric_df[col].values

    # 5. Build Visualization
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Cluster',
        title=f"KikoLens Intelligence: Data Segmentation ({n_clusters} Clusters)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_data=numeric_df.columns[:3].tolist()
    )
    
    # 6. Calculate Cluster Profiles (Averages)
    df_with_clusters = df.copy()
    df_with_clusters['Kiko_Cluster'] = clusters
    profiles = df_with_clusters.groupby('Kiko_Cluster')[numeric_df.columns].mean()
    
    return {
        "fig": fig,
        "profiles": profiles,
        "n_clusters": n_clusters,
        "df_with_clusters": df_with_clusters
    }
