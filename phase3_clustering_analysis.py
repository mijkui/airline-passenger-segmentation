#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusteringAnalysis:
    """Comprehensive clustering analysis pipeline for airline passenger satisfaction data."""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.feature_names = None
        self.pc_names = None
        self.clustering_results = {}
        self.best_k = None
        self.best_algorithm = None
        self.cluster_labels = None
        
    def load_processed_data(self, train_path, test_path, pca_train_path, pca_test_path):
        """Load processed data from Phase 1 and PCA results from Phase 2."""
        print("Loading processed data from Phase 1 and PCA results from Phase 2...")
        
        # Load original processed data
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Load PCA transformed data
        self.X_train_pca = np.loadtxt(pca_train_path, delimiter=',')
        self.X_test_pca = np.loadtxt(pca_test_path, delimiter=',')
        
        # Separate features and target
        feature_cols = [col for col in self.train_df.columns if col != 'satisfaction']
        self.X_train = self.train_df[feature_cols].values
        self.X_test = self.test_df[feature_cols].values
        self.y_train = self.train_df['satisfaction'].values
        self.y_test = self.test_df['satisfaction'].values
        
        # Set feature names
        self.feature_names = feature_cols
        self.pc_names = [f'PC{i+1}' for i in range(self.X_train_pca.shape[1])]
        
        print(f"Original features shape: {self.X_train.shape}")
        print(f"PCA features shape: {self.X_train_pca.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        
        return self
    
    def choose_feature_space(self):
        """Choose feature space for clustering analysis."""
        print("\nChoosing feature space")
        
        # Option 1: Use top K PCs (recommended)
        print("Option 1: Using PCA features (recommended)")
        print(f"  - Shape: {self.X_train_pca.shape}")
        print(f"  - Advantages: Reduced noise, no collinearity, interpretable")
        
        # Option 2: Use original standardized features
        print("\nOption 2: Using original standardized features")
        print(f"  - Shape: {self.X_train.shape}")
        print(f"  - Advantages: All information preserved")
        print(f"  - Disadvantages: Higher dimensionality, potential collinearity")
        
        # For this analysis, we'll use both and compare
        self.feature_spaces = {
            'PCA': self.X_train_pca,
            'Original': self.X_train
        }
        
        return self
    
    def run_kmeans_analysis(self, X, feature_space_name):
        """Run K-Means clustering for K=2..10."""
        print(f"\nRunning K-Means on {feature_space_name.upper()} features")
        
        k_range = range(2, 11)
        results = {
            'k_values': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'models': []
        }
        
        for k in k_range:
            print(f"  Testing K={k}...")
            
            # Run K-Means with multiple initializations
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, cluster_labels)
            calinski = calinski_harabasz_score(X, cluster_labels)
            davies_bouldin = davies_bouldin_score(X, cluster_labels)
            
            # Store results
            results['k_values'].append(k)
            results['inertia'].append(inertia)
            results['silhouette'].append(silhouette)
            results['calinski_harabasz'].append(calinski)
            results['davies_bouldin'].append(davies_bouldin)
            results['models'].append(kmeans)
            
            print(f"    Silhouette: {silhouette:.3f}, Inertia: {inertia:.0f}")
        
        # Find best K based on silhouette score
        best_idx = np.argmax(results['silhouette'])
        best_k = results['k_values'][best_idx]
        best_model = results['models'][best_idx]
        
        print(f"\nBest K for {feature_space_name}: {best_k} (Silhouette: {results['silhouette'][best_idx]:.3f})")
        
        return results, best_k, best_model
    
    def run_gmm_analysis(self, X, feature_space_name):
        """Run Gaussian Mixture Models with different covariance types."""
        print(f"\nRunning GMM on {feature_space_name.upper()} features")
        
        k_range = range(2, 11)
        covariance_types = ['diag', 'full']
        results = {
            'k_values': [],
            'covariance_type': [],
            'aic': [],
            'bic': [],
            'silhouette': [],
            'models': []
        }
        
        for k in k_range:
            for cov_type in covariance_types:
                print(f"  Testing K={k}, Covariance={cov_type}...")
                
                # Run GMM
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
                cluster_labels = gmm.fit_predict(X)
                
                # Calculate metrics
                aic = gmm.aic(X)
                bic = gmm.bic(X)
                silhouette = silhouette_score(X, cluster_labels)
                
                # Store results
                results['k_values'].append(k)
                results['covariance_type'].append(cov_type)
                results['aic'].append(aic)
                results['bic'].append(bic)
                results['silhouette'].append(silhouette)
                results['models'].append(gmm)
                
                print(f"    AIC: {aic:.0f}, BIC: {bic:.0f}, Silhouette: {silhouette:.3f}")
        
        # Find best model based on BIC
        best_idx = np.argmin(results['bic'])
        best_k = results['k_values'][best_idx]
        best_cov = results['covariance_type'][best_idx]
        best_model = results['models'][best_idx]
        
        print(f"\nBest GMM for {feature_space_name}: K={best_k}, Covariance={best_cov} (BIC: {results['bic'][best_idx]:.0f})")
        
        return results, best_k, best_model
    
    def run_hierarchical_analysis(self, X, feature_space_name):
        """Run Hierarchical clustering with Ward linkage."""
        print(f"\nRunning hierarchical clustering on {feature_space_name.upper()} features")
        
        k_range = range(2, 11)
        results = {
            'k_values': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'models': []
        }
        
        for k in k_range:
            print(f"  Testing K={k}...")
            
            # Run Hierarchical clustering
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
            cluster_labels = hierarchical.fit_predict(X)
            
            # Calculate metrics
            silhouette = silhouette_score(X, cluster_labels)
            calinski = calinski_harabasz_score(X, cluster_labels)
            davies_bouldin = davies_bouldin_score(X, cluster_labels)
            
            # Store results
            results['k_values'].append(k)
            results['silhouette'].append(silhouette)
            results['calinski_harabasz'].append(calinski)
            results['davies_bouldin'].append(davies_bouldin)
            results['models'].append(hierarchical)
            
            print(f"    Silhouette: {silhouette:.3f}")
        
        # Find best K based on silhouette score
        best_idx = np.argmax(results['silhouette'])
        best_k = results['k_values'][best_idx]
        best_model = results['models'][best_idx]
        
        print(f"\nBest K for Hierarchical {feature_space_name}: {best_k} (Silhouette: {results['silhouette'][best_idx]:.3f})")
        
        return results, best_k, best_model
    
    def run_dbscan_analysis(self, X, feature_space_name):
        """Run DBSCAN clustering."""
        print(f"\nRunning dbscan on {feature_space_name.upper()} features")
        
        # Try different eps values
        eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        min_samples_values = [5, 10, 15, 20]
        
        results = {
            'eps': [],
            'min_samples': [],
            'n_clusters': [],
            'silhouette': [],
            'models': []
        }
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                print(f"  Testing eps={eps}, min_samples={min_samples}...")
                
                # Run DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(X)
                
                # Count clusters (excluding noise points labeled as -1)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters > 1:  # Only calculate metrics if we have clusters
                    # Calculate silhouette score (excluding noise points)
                    if len(set(cluster_labels)) > 1:
                        silhouette = silhouette_score(X[cluster_labels != -1], cluster_labels[cluster_labels != -1])
                    else:
                        silhouette = -1
                else:
                    silhouette = -1
                
                # Store results
                results['eps'].append(eps)
                results['min_samples'].append(min_samples)
                results['n_clusters'].append(n_clusters)
                results['silhouette'].append(silhouette)
                results['models'].append(dbscan)
                
                print(f"    Clusters: {n_clusters}, Silhouette: {silhouette:.3f}")
        
        # Find best parameters
        valid_results = [(i, results['silhouette'][i]) for i in range(len(results['silhouette'])) if results['silhouette'][i] > 0]
        if valid_results:
            best_idx = max(valid_results, key=lambda x: x[1])[0]
            best_eps = results['eps'][best_idx]
            best_min_samples = results['min_samples'][best_idx]
            best_model = results['models'][best_idx]
            print(f"\nBest DBSCAN for {feature_space_name}: eps={best_eps}, min_samples={best_min_samples}")
        else:
            print(f"\nNo valid DBSCAN clusters found for {feature_space_name}")
            best_model = None
        
        return results, best_model
    
    def create_elbow_plot(self, kmeans_results, feature_space_name):
        """Create elbow plot for K-Means results."""
        print(f"\nCreating elbow plot for {feature_space_name.upper()}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow plot
        ax1.plot(kmeans_results['k_values'], kmeans_results['inertia'], 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title(f'Elbow Plot - {feature_space_name} Features')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(kmeans_results['k_values'], kmeans_results['silhouette'], 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title(f'Silhouette Plot - {feature_space_name} Features')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best K
        best_k_idx = np.argmax(kmeans_results['silhouette'])
        best_k = kmeans_results['k_values'][best_k_idx]
        ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
        ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'elbow_plot_{feature_space_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_gmm_plots(self, gmm_results, feature_space_name):
        """Create plots for GMM results."""
        print(f"\nCreating GMM plots for {feature_space_name.upper()}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BIC plot
        k_values = sorted(set(gmm_results['k_values']))
        bic_diag = [min([gmm_results['bic'][i] for i in range(len(gmm_results['k_values'])) 
                        if gmm_results['k_values'][i] == k and gmm_results['covariance_type'][i] == 'diag']) 
                   for k in k_values]
        bic_full = [min([gmm_results['bic'][i] for i in range(len(gmm_results['k_values'])) 
                        if gmm_results['k_values'][i] == k and gmm_results['covariance_type'][i] == 'full']) 
                   for k in k_values]
        
        ax1.plot(k_values, bic_diag, 'bo-', label='Diagonal', linewidth=2, markersize=6)
        ax1.plot(k_values, bic_full, 'ro-', label='Full', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('BIC')
        ax1.set_title(f'BIC Plot - {feature_space_name} Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AIC plot
        aic_diag = [min([gmm_results['aic'][i] for i in range(len(gmm_results['k_values'])) 
                        if gmm_results['k_values'][i] == k and gmm_results['covariance_type'][i] == 'diag']) 
                   for k in k_values]
        aic_full = [min([gmm_results['aic'][i] for i in range(len(gmm_results['k_values'])) 
                        if gmm_results['k_values'][i] == k and gmm_results['covariance_type'][i] == 'full']) 
                   for k in k_values]
        
        ax2.plot(k_values, aic_diag, 'bo-', label='Diagonal', linewidth=2, markersize=6)
        ax2.plot(k_values, aic_full, 'ro-', label='Full', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('AIC')
        ax2.set_title(f'AIC Plot - {feature_space_name} Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'gmm_plots_{feature_space_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def stability_analysis(self, X, model, feature_space_name, n_bootstrap=50):
        """Perform stability analysis using bootstrap resampling."""
        print(f"\nStablility Analysis for {feature_space_name.upper()}")
        
        # Get original cluster assignments
        original_labels = model.fit_predict(X)
        
        # Bootstrap resampling
        silhouette_scores = []
        jaccard_similarities = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_bootstrap = X[indices]
            
            # Fit model on bootstrap sample
            model_bootstrap = type(model)(**model.get_params())
            bootstrap_labels = model_bootstrap.fit_predict(X_bootstrap)
            
            # Calculate silhouette score
            if len(set(bootstrap_labels)) > 1:
                silhouette = silhouette_score(X_bootstrap, bootstrap_labels)
                silhouette_scores.append(silhouette)
            
            # Calculate Jaccard similarity (simplified)
            # This is a simplified version - in practice, you'd need to match clusters
            if len(set(bootstrap_labels)) == len(set(original_labels)):
                jaccard_sim = len(set(original_labels) & set(bootstrap_labels)) / len(set(original_labels) | set(bootstrap_labels))
                jaccard_similarities.append(jaccard_sim)
        
        # Calculate stability metrics
        mean_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
        std_silhouette = np.std(silhouette_scores) if silhouette_scores else 0
        mean_jaccard = np.mean(jaccard_similarities) if jaccard_similarities else 0
        
        print(f"  Mean Silhouette: {mean_silhouette:.3f} ± {std_silhouette:.3f}")
        print(f"  Mean Jaccard Similarity: {mean_jaccard:.3f}")
        
        return {
            'mean_silhouette': mean_silhouette,
            'std_silhouette': std_silhouette,
            'mean_jaccard': mean_jaccard
        }
    
    def separability_test(self, X, cluster_labels, feature_space_name):
        """Test cluster separability using supervised learning."""
        print(f"\nSeperability test for {feature_space_name.upper()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, cluster_labels, test_size=0.3, random_state=42)
        
        # Test with Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Test with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        print(f"  Logistic Regression Accuracy: {lr_accuracy:.3f}")
        print(f"  Random Forest Accuracy: {rf_accuracy:.3f}")
        
        # High accuracy indicates good separability
        if lr_accuracy > 0.8 or rf_accuracy > 0.8:
            print("  ✅ Clusters are well-separated")
        else:
            print("  ⚠️  Clusters may not be well-separated")
        
        return {
            'lr_accuracy': lr_accuracy,
            'rf_accuracy': rf_accuracy
        }
    
    def profile_clusters(self, X, cluster_labels, feature_names, feature_space_name):
        """Profile clusters by analyzing their characteristics."""
        print(f"\nCluster profiting for {feature_space_name.upper()}")
        
        # Create DataFrame with cluster labels
        if feature_space_name == 'PCA':
            df = pd.DataFrame(X, columns=self.pc_names)
        else:
            df = pd.DataFrame(X, columns=feature_names)
        
        df['cluster'] = cluster_labels
        n_clusters = len(set(cluster_labels))
        
        print(f"Number of clusters: {n_clusters}")
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(df)) * 100
            
            print(f"\nCluster {cluster_id}: {cluster_size} samples ({cluster_pct:.1f}%)")
            
            # Calculate means for key features
            if feature_space_name == 'PCA':
                # For PCA features, show top loadings
                means = cluster_data[self.pc_names].mean()
                print("  Top PC means:")
                for pc in self.pc_names:
                    print(f"    {pc}: {means[pc]:.3f}")
            else:
                # For original features, show key features
                key_features = ['Age', 'Flight Distance', 'Inflight wifi service', 
                              'Seat comfort', 'Food and drink', 'Departure Delay in Minutes']
                available_features = [f for f in key_features if f in df.columns]
                
                if available_features:
                    means = cluster_data[available_features].mean()
                    print("  Key feature means:")
                    for feature in available_features:
                        print(f"    {feature}: {means[feature]:.3f}")
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_pct,
                'means': means.to_dict() if 'means' in locals() else {}
            })
        
        # Statistical tests for significant differences
        if feature_space_name == 'Original':
            print(f"\nStatistical tests")
            for feature in available_features:
                if feature in df.columns:
                    groups = [df[df['cluster'] == i][feature].values for i in range(n_clusters)]
                    f_stat, p_value = stats.f_oneway(*groups)
                    print(f"{feature}: F={f_stat:.3f}, p={p_value:.3f}")
        
        return cluster_stats
    
    def name_clusters(self, cluster_stats, feature_space_name):
        """Name clusters based on their characteristics."""
        print(f"\nCluster naming for {feature_space_name.upper()}")
        
        cluster_names = {}
        
        for stats in cluster_stats:
            cluster_id = stats['cluster_id']
            means = stats['means']
            
            if feature_space_name == 'PCA':
                # Name based on PC characteristics
                if 'PC1' in means and means['PC1'] > 0:
                    cluster_names[cluster_id] = "High Delay Risk"
                elif 'PC2' in means and means['PC2'] > 0:
                    cluster_names[cluster_id] = "Service Quality Focused"
                elif 'PC3' in means and means['PC3'] > 0:
                    cluster_names[cluster_id] = "Digital Convenience Seekers"
                else:
                    cluster_names[cluster_id] = f"General Cluster {cluster_id}"
            else:
                # Name based on original feature characteristics
                if 'Age' in means and means['Age'] > 0:
                    cluster_names[cluster_id] = "Mature Travelers"
                elif 'Flight Distance' in means and means['Flight Distance'] > 0:
                    cluster_names[cluster_id] = "Long-haul Travelers"
                else:
                    cluster_names[cluster_id] = f"Mixed Cluster {cluster_id}"
            
            print(f"Cluster {cluster_id}: {cluster_names[cluster_id]}")
        
        return cluster_names
    
    def create_cluster_visualizations(self, X, cluster_labels, feature_space_name, cluster_names):
        """Create comprehensive cluster visualizations."""
        print(f"\nCreating cluster visualizations for {feature_space_name.upper()}")
        
        if X.shape[1] < 2:
            print("Cannot create 2D visualizations - need at least 2 features")
            return
        
        # Use first two components/features for visualization
        X_2d = X[:, :2]
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}: {cluster_names.get(label, "Unknown")}', 
                       alpha=0.7, s=50)
        
        # Plot centroids
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            centroid = np.mean(X_2d[mask], axis=0)
            plt.scatter(centroid[0], centroid[1], c=[colors[i]], marker='x', s=200, linewidths=3)
        
        plt.xlabel(f'{self.pc_names[0] if feature_space_name == "PCA" else "Feature 1"}')
        plt.ylabel(f'{self.pc_names[1] if feature_space_name == "PCA" else "Feature 2"}')
        plt.title(f'Cluster Visualization - {feature_space_name} Features')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'cluster_visualization_{feature_space_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, train_path, test_path, pca_train_path, pca_test_path):
        """Run the complete Phase 3 clustering analysis."""
        print("="*60)
        print("PHASE 3 - CLUSTERING ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_processed_data(train_path, test_path, pca_train_path, pca_test_path)
        
        # Choose feature spaces
        self.choose_feature_space()
        
        # Run clustering on both feature spaces
        for feature_space_name, X in self.feature_spaces.items():
            print(f"\n{'='*50}")
            print(f"ANALYZING {feature_space_name.upper()} FEATURES")
            print(f"{'='*50}")
            
            # Run K-Means
            kmeans_results, best_k_kmeans, best_kmeans = self.run_kmeans_analysis(X, feature_space_name)
            self.create_elbow_plot(kmeans_results, feature_space_name)
            
            # Run GMM
            gmm_results, best_k_gmm, best_gmm = self.run_gmm_analysis(X, feature_space_name)
            self.create_gmm_plots(gmm_results, feature_space_name)
            
            # Run Hierarchical
            hierarchical_results, best_k_hier, best_hier = self.run_hierarchical_analysis(X, feature_space_name)
            
            # Run DBSCAN
            dbscan_results, best_dbscan = self.run_dbscan_analysis(X, feature_space_name)
            
            # Choose best algorithm based on silhouette score
            algorithms = {
                'K-Means': (best_kmeans, kmeans_results['silhouette'][best_k_kmeans-2]),
                'GMM': (best_gmm, gmm_results['silhouette'][np.argmin(gmm_results['bic'])]),
                'Hierarchical': (best_hier, hierarchical_results['silhouette'][best_k_hier-2])
            }
            
            if best_dbscan is not None:
                dbscan_labels = best_dbscan.fit_predict(X)
                if len(set(dbscan_labels)) > 1:
                    dbscan_silhouette = silhouette_score(X[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
                    algorithms['DBSCAN'] = (best_dbscan, dbscan_silhouette)
            
            # Select best algorithm
            best_algorithm_name = max(algorithms.keys(), key=lambda k: algorithms[k][1])
            best_algorithm = algorithms[best_algorithm_name][0]
            best_silhouette = algorithms[best_algorithm_name][1]
            
            print(f"\nBest algorithm for {feature_space_name}: {best_algorithm_name} (Silhouette: {best_silhouette:.3f})")
            
            # Get cluster labels
            cluster_labels = best_algorithm.fit_predict(X)
            
            # Stability analysis
            stability_results = self.stability_analysis(X, best_algorithm, feature_space_name)
            
            # Separability test
            separability_results = self.separability_test(X, cluster_labels, feature_space_name)
            
            # Profile clusters
            cluster_stats = self.profile_clusters(X, cluster_labels, self.feature_names, feature_space_name)
            
            # Name clusters
            cluster_names = self.name_clusters(cluster_stats, feature_space_name)
            
            # Create visualizations
            self.create_cluster_visualizations(X, cluster_labels, feature_space_name, cluster_names)
            
            # Store results
            self.clustering_results[feature_space_name] = {
                'algorithm': best_algorithm_name,
                'model': best_algorithm,
                'labels': cluster_labels,
                'silhouette': best_silhouette,
                'stability': stability_results,
                'separability': separability_results,
                'cluster_stats': cluster_stats,
                'cluster_names': cluster_names
            }
        
        print("\n" + "="*60)
        print("PHASE 3 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self

def main():
    """Main function to run the clustering analysis."""
    # Initialize clustering analysis
    clustering = ClusteringAnalysis()
    
    # Run complete analysis
    clustering.run_complete_analysis(
        train_path='train_processed.csv',
        test_path='test_processed.csv',
        pca_train_path='X_train_pca.csv',
        pca_test_path='X_test_pca.csv'
    )

if __name__ == "__main__":
    main()
