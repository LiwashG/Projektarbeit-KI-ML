
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
import umap
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN

# ==============================================================================
# 1. data preparation and feature importance
# ==============================================================================

# load data
df = pd.read_excel("chiefs_knife_dataset.xlsx")

# set label for specification limit
df['label'] = df['Ra'].apply(
    lambda x: 'in specification limit' if 0.13 <= x <= 0.21
    else 'out of specification limit')

# prepare features
X = df.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss", "label"])
Y = X.loc[:, X.nunique(dropna=False) != 1]

# --- feature scaling  ---
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = Normalizer()
Y_scaled = pd.DataFrame(scaler.fit_transform(Y), columns=Y.columns, index=Y.index)

# --- Feature Importance with Random Forest ---
target_variable_Ra = df['Ra']
rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf_model.fit(Y_scaled, target_variable_Ra)

feature_importances = pd.Series(rf_model.feature_importances_, index=Y_scaled.columns)
feature_importances = feature_importances.sort_values(ascending=False)

N = 10
top_features = feature_importances.head(N).index.tolist()

print("--- Top 10 Features for the analysis ---")
print(top_features)
print("="*40 + "\n")


# ==============================================================================
# 2. UMAP Grid-Search for visual exploration
# ==============================================================================

print("--- performing UMAP Grid-Search... ---")
metrics = ['euclidean', 'manhattan', 'cosine']
n_neighbors_list = [5, 15, 50]
min_dist_list = [0.0, 0.1, 0.5]

cols = len(min_dist_list)
rows = len(metrics) * len(n_neighbors_list)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
fig.suptitle(f'UMAP Grid-Search with Top {N} features', fontsize=16)

for i, metric in enumerate(metrics):
    for j, n_neighbors in enumerate(n_neighbors_list):
        for k, min_dist in enumerate(min_dist_list):
            ax = axes[i*len(n_neighbors_list) + j, k]
            reducer = umap.UMAP(n_components=2, metric=metric, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            embedding = reducer.fit_transform(Y_scaled[top_features])
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=(df['label'] == 'in specification limit'), cmap='coolwarm', s=5, alpha=0.8)
            ax.set_title(f"metric={metric}, n={n_neighbors}, min={min_dist}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ==============================================================================
# 3. DBSCAN-analysis for specific UMAP result
# ==============================================================================

print("\n--- perform DBSCAN analysis for specific parameters... ---")
# --- create UMAP-embedding for the analysis ---
reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=50, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(Y_scaled[top_features])

# --- perform DBSCAN-algorithm ---
dbscan = DBSCAN(eps=0.45, min_samples=10)
clusters = dbscan.fit_predict(embedding)
df['cluster'] = clusters

# --- statistical analysis of the cluster ---
is_blue = df['label'] == 'out of specification limit'
is_red = df['label'] == 'in specification limit'
is_in_target_clusters = df['cluster'].isin([1, 2])

total_blue_points = df[is_blue].shape[0]
total_red_points = df[is_red].shape[0]
blue_points_in_target_clusters = df[is_blue & is_in_target_clusters].shape[0]
red_points_in_target_clusters = df[is_red & is_in_target_clusters].shape[0]
total_points_in_target_clusters = df[is_in_target_clusters].shape[0]

percentage_blue = (blue_points_in_target_clusters / total_blue_points) * 100 if total_blue_points > 0 else 0
percentage_red = (red_points_in_target_clusters / total_red_points) * 100 if total_blue_points > 0 else 0
percentage_of_red_in_clusters = (red_points_in_target_clusters / total_points_in_target_clusters) * 100 if total_points_in_target_clusters > 0 else 0
percentage_of_blue_in_clusters = (blue_points_in_target_clusters / total_points_in_target_clusters) * 100 if total_points_in_target_clusters > 0 else 0

print("\n--- global analysis ---")
print(f"Percentage of all blue scatters in clusters 1 & 2: {percentage_blue:.2f}%")
print(f"Percentage of all red scatters in clusters 1 & 2: {percentage_red:.2f}%")
print("\n--- analysis of composition IN clusters 1 & 2 ---")
print(f"Percentage of red scatters of all scatters in clusters 1 & 2: {percentage_of_red_in_clusters:.2f}%")
print(f"In total there are {red_points_in_target_clusters} out of {total_red_points} red scatters in clusters 1 & 2.")
print(f"Percentage of blue scatters of all scatters in clusters 1 & 2: {percentage_of_blue_in_clusters:.2f}%")
print(f"In total there are {blue_points_in_target_clusters} out of {total_blue_points} red scatters in clusters 1 & 2.")

# --- visualization of DBSCAN-Cluster ---
plt.figure(figsize=(10, 8))
unique_labels = set(clusters)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1: col = [0, 0, 0, 1]
    class_member_mask = (clusters == k)
    xy = embedding[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6 if k != -1 else 3, label=f'cluster {k}' if k != -1 else 'noise')

title_text = (f'DBSCAN Clustering\n'
              f'{percentage_blue:.2f}% of all blue scatters are in clusters 1 & 2\n'
              f'cluster 1 & 2 consist of {percentage_of_red_in_clusters:.2f}% red scatters')
plt.title(title_text)
plt.legend()
plt.show()