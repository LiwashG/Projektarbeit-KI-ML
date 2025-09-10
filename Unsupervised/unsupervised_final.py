import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
import umap
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN


class KnifeSurfaceAnalyzer:
    """
    Class for analyzing knife surface data using feature importance,
    dimensionality reduction (UMAP), and clustering (DBSCAN).
    """

    def __init__(self, filepath, top_n_features=10, random_state=42):
        """
        Initialize the analyzer with dataset path and parameters.

        Parameters
        ----------
        filepath : str
            Path to the input Excel dataset.
        top_n_features : int, optional
            Number of top features to be selected for analysis (default is 10).
        random_state : int, optional
            Random seed for reproducibility (default is 42).
        """
        self.filepath = filepath
        self.top_n_features = top_n_features
        self.random_state = random_state

        # Data containers
        self.df = None
        self.Y_scaled = None
        self.top_features = None
        self.embedding = None
        self.clusters = None

    # 1. Data Preparation and Feature Importance
    def load_and_prepare_data(self):
        """
        Load dataset, preprocess features, scale them,
        and compute feature importance using Random Forest.
        """
        print("--- Loading and preparing data ---")

        # Load dataset
        self.df = pd.read_excel(self.filepath)

        # Create specification limit label based on Ra
        self.df['label'] = self.df['Ra'].apply(
            lambda x: 'in specification limit' if 0.13 <= x <= 0.21
            else 'out of specification limit')

        # Select valid features (drop non-feature columns and constant columns)
        X = self.df.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss", "label"])
        Y = X.loc[:, X.nunique(dropna=False) != 1]

        # Scale features using RobustScaler
        scaler = RobustScaler()
        self.Y_scaled = pd.DataFrame(scaler.fit_transform(Y), columns=Y.columns, index=Y.index)

        # Train Random Forest Regressor for feature importance
        target_variable_Ra = self.df['Ra']
        rf_model = RandomForestRegressor(n_estimators=150, random_state=self.random_state, n_jobs=-1)
        rf_model.fit(self.Y_scaled, target_variable_Ra)

        # Rank features by importance
        feature_importances = pd.Series(rf_model.feature_importances_, index=self.Y_scaled.columns)
        feature_importances = feature_importances.sort_values(ascending=False)

        # Select top N features
        self.top_features = feature_importances.head(self.top_n_features).index.tolist()

        print(f"--- Top {self.top_n_features} Features for the analysis ---")
        print(self.top_features)
        print("=" * 40 + "\n")

    # 2. UMAP Grid-Search Visualization
    def umap_grid_search(self, metrics=None, n_neighbors_list=None, min_dist_list=None):
        """
        Perform UMAP dimensionality reduction with different parameter
        combinations and visualize embeddings in a grid layout.

        Parameters
        ----------
        metrics : list of str, optional
            List of distance metrics to test (default: euclidean, manhattan, cosine).
        n_neighbors_list : list of int, optional
            List of neighbor values to test (default: [5, 15, 50]).
        min_dist_list : list of float, optional
            List of minimum distance values to test (default: [0.0, 0.1, 0.5]).
        """
        print("--- Performing UMAP Grid-Search ---")

        if metrics is None:
            metrics = ['euclidean', 'manhattan', 'cosine']
        if n_neighbors_list is None:
            n_neighbors_list = [5, 15, 50]
        if min_dist_list is None:
            min_dist_list = [0.0, 0.1, 0.5]

        cols = len(min_dist_list)
        rows = len(metrics) * len(n_neighbors_list)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(f'UMAP Grid-Search with Top {self.top_n_features} features', fontsize=16)

        # Iterate over parameter combinations
        for i, metric in enumerate(metrics):
            for j, n_neighbors in enumerate(n_neighbors_list):
                for k, min_dist in enumerate(min_dist_list):
                    ax = axes[i * len(n_neighbors_list) + j, k]

                    # Compute UMAP embedding
                    reducer = umap.UMAP(
                        n_components=2, metric=metric, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=self.random_state
                    )
                    embedding = reducer.fit_transform(self.Y_scaled[self.top_features])

                    # Scatter plot
                    ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=(self.df['label'] == 'in specification limit'),
                               cmap='coolwarm', s=5, alpha=0.8)
                    ax.set_title(f"metric={metric}, n={n_neighbors}, min={min_dist}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 3. DBSCAN Analysis
    def dbscan_analysis(self, metric='cosine', n_neighbors=50, min_dist=0.1,
                        eps=0.45, min_samples=10, target_clusters=(1, 2)):
        """
        Apply UMAP for dimensionality reduction followed by DBSCAN clustering.
        Perform statistical analysis of clusters and visualize results.

        Parameters
        ----------
        metric : str, optional
            Distance metric for UMAP (default: 'cosine').
        n_neighbors : int, optional
            Number of neighbors for UMAP (default: 50).
        min_dist : float, optional
            Minimum distance for UMAP (default: 0.1).
        eps : float, optional
            Maximum distance between points in DBSCAN (default: 0.45).
        min_samples : int, optional
            Minimum number of samples in a DBSCAN cluster (default: 10).
        target_clusters : tuple of int, optional
            Clusters of interest for statistical analysis (default: (1, 2)).
        """
        print("\n--- Performing DBSCAN Analysis ---")

        # Create UMAP embedding
        reducer = umap.UMAP(
            n_components=2, metric=metric, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=self.random_state
        )
        self.embedding = reducer.fit_transform(self.Y_scaled[self.top_features])

        # Run DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.embedding)
        self.df['cluster'] = self.clusters

        # Cluster statistics
        is_blue = self.df['label'] == 'out of specification limit'
        is_red = self.df['label'] == 'in specification limit'
        is_in_target_clusters = self.df['cluster'].isin(target_clusters)

        total_blue_points = self.df[is_blue].shape[0]
        total_red_points = self.df[is_red].shape[0]
        blue_points_in_target_clusters = self.df[is_blue & is_in_target_clusters].shape[0]
        red_points_in_target_clusters = self.df[is_red & is_in_target_clusters].shape[0]
        total_points_in_target_clusters = self.df[is_in_target_clusters].shape[0]

        # Compute percentages
        percentage_blue = (blue_points_in_target_clusters / total_blue_points) * 100 if total_blue_points > 0 else 0
        percentage_red = (red_points_in_target_clusters / total_red_points) * 100 if total_red_points > 0 else 0
        percentage_of_red_in_clusters = (red_points_in_target_clusters / total_points_in_target_clusters) * 100 if total_points_in_target_clusters > 0 else 0
        percentage_of_blue_in_clusters = (blue_points_in_target_clusters / total_points_in_target_clusters) * 100 if total_points_in_target_clusters > 0 else 0

        # Print analysis
        print("\n--- Global analysis ---")
        print(f"Percentage of all blue samples in clusters {target_clusters}: {percentage_blue:.2f}%")
        print(f"Percentage of all red samples in clusters {target_clusters}: {percentage_red:.2f}%")
        print("\n--- Composition of target clusters ---")
        print(f"Percentage of red samples inside target clusters: {percentage_of_red_in_clusters:.2f}%")
        print(f"Count: {red_points_in_target_clusters} out of {total_red_points} red samples.")
        print(f"Percentage of blue samples inside target clusters: {percentage_of_blue_in_clusters:.2f}%")
        print(f"Count: {blue_points_in_target_clusters} out of {total_blue_points} blue samples.")

        # Visualize DBSCAN clusters
        plt.figure(figsize=(10, 8))
        unique_labels = set(self.clusters)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:  # Noise
                col = [0, 0, 0, 1]
            class_member_mask = (self.clusters == k)
            xy = self.embedding[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',
                     markerfacecolor=tuple(col),
                     markeredgecolor='k',
                     markersize=6 if k != -1 else 3,
                     label=f'cluster {k}' if k != -1 else 'noise')

        title_text = (f'DBSCAN Clustering\n'
                      f'{percentage_blue:.2f}% of blue samples in clusters {target_clusters}\n'
                      f'{percentage_of_red_in_clusters:.2f}% of samples in target clusters are red')
        plt.title(title_text)
        plt.legend()
        plt.show()



# Example usage
if __name__ == "__main__":
    analyzer = KnifeSurfaceAnalyzer("chiefs_knife_dataset.xlsx", top_n_features=10, random_state=42)
    analyzer.load_and_prepare_data()
    analyzer.umap_grid_search()
    analyzer.dbscan_analysis()
