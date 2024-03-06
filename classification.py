import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your dataset
df = pd.read_csv('C:/Users/rachel/Downloads/apple_quality.csv')

# Drop any non-numeric columns for simplicity
df_numeric = df.select_dtypes(include=['number'])

# Fill any missing values
df_numeric.fillna(df_numeric.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_numeric)

# Perform t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_features)

# Compute silhouette score for each column
silhouette_scores = []
for col in df_numeric.columns:
    target = df_numeric[col]
    silhouette_scores.append(silhouette_score(tsne_results, target))

# Find the column with the highest silhouette score
most_potential_target_column = df_numeric.columns[np.argmax(silhouette_scores)]

print("Most Potential Target Column:", most_potential_target_column)
