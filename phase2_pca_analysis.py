#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PCAAnalysis:
    """Comprehensive PCA analysis pipeline for airline passenger satisfaction data."""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pca = None
        self.n_components = None
        self.loadings = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        
    def load_processed_data(self, train_path, test_path):
        """Load processed data from Phase 1."""
        print("Loading processed data from Phase 1...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Separate features and target
        feature_cols = [col for col in self.train_df.columns if col != 'satisfaction']
        self.X_train = self.train_df[feature_cols].values
        self.X_test = self.test_df[feature_cols].values
        self.y_train = self.train_df['satisfaction'].values
        self.y_test = self.test_df['satisfaction'].values
        
        print(f"Train features shape: {self.X_train.shape}")
        print(f"Test features shape: {self.X_test.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        
        return self
    
    def fit_pca(self, n_components=None):
        """Fit PCA on standardized features."""
        print(f"\n=== FITTING PCA ===")
        
        # Fit PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(self.X_train)
        
        # Get explained variance
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        print(f"PCA fitted with {self.pca.n_components_} components")
        print(f"Total explained variance: {self.cumulative_variance[-1]:.3f}")
        
        return self
    
    def create_scree_plot(self):
        """Create scree plot and cumulative explained variance plot."""
        print("\n=== CREATING SCREE PLOT ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scree plot
        components = range(1, len(self.explained_variance_ratio) + 1)
        ax1.plot(components, self.explained_variance_ratio, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add elbow point detection
        diffs = np.diff(self.explained_variance_ratio)
        diffs2 = np.diff(diffs)
        elbow_point = np.argmax(diffs2) + 2  # +2 because of double diff
        ax1.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7, 
                   label=f'Elbow at PC{elbow_point}')
        ax1.legend()
        
        # Cumulative explained variance
        ax2.plot(components, self.cumulative_variance, 'ro-', linewidth=2, markersize=6)
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Find optimal number of components
        n_80 = np.where(self.cumulative_variance >= 0.8)[0][0] + 1
        n_95 = np.where(self.cumulative_variance >= 0.95)[0][0] + 1
        
        ax2.axvline(x=n_80, color='green', linestyle=':', alpha=0.7)
        ax2.axvline(x=n_95, color='orange', linestyle=':', alpha=0.7)
        
        print(f"Components for 80% variance: {n_80}")
        print(f"Components for 95% variance: {n_95}")
        print(f"Elbow point: {elbow_point}")
        
        plt.tight_layout()
        plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return n_80, n_95, elbow_point
    
    def parallel_analysis(self, n_iterations=100):
        """Perform parallel analysis to determine optimal number of components."""
        print("\n=== PARALLEL ANALYSIS ===")
        
        n_samples, n_features = self.X_train.shape
        random_eigenvalues = []
        
        # Generate random data with same dimensions
        for i in range(n_iterations):
            random_data = np.random.normal(0, 1, (n_samples, n_features))
            random_pca = PCA()
            random_pca.fit(random_data)
            random_eigenvalues.append(random_pca.explained_variance_ratio_)
        
        # Calculate mean random eigenvalues
        mean_random_eigenvalues = np.mean(random_eigenvalues, axis=0)
        
        # Find where real eigenvalues exceed random
        real_eigenvalues = self.explained_variance_ratio
        n_components_parallel = np.where(real_eigenvalues > mean_random_eigenvalues)[0]
        
        if len(n_components_parallel) > 0:
            n_components_parallel = n_components_parallel[-1] + 1
        else:
            n_components_parallel = 1
        
        # Plot parallel analysis
        plt.figure(figsize=(10, 6))
        components = range(1, len(real_eigenvalues) + 1)
        plt.plot(components, real_eigenvalues, 'bo-', label='Real Data', linewidth=2)
        plt.plot(components, mean_random_eigenvalues, 'ro-', label='Random Data', linewidth=2)
        plt.axvline(x=n_components_parallel, color='green', linestyle='--', 
                   label=f'Parallel Analysis: {n_components_parallel} components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Parallel Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('parallel_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Parallel analysis suggests: {n_components_parallel} components")
        
        return n_components_parallel
    
    def determine_optimal_components(self):
        """Determine optimal number of components using multiple methods."""
        print("\n=== DETERMINING OPTIMAL COMPONENTS ===")
        
        # First fit PCA with all components to get scree plot
        self.fit_pca()
        
        # Get recommendations from different methods
        n_80, n_95, elbow = self.create_scree_plot()
        n_parallel = self.parallel_analysis()
        
        # Choose optimal number (conservative approach)
        self.n_components = min(n_80, n_parallel)
        
        print(f"\nComponent selection summary:")
        print(f"  80% variance: {n_80} components")
        print(f"  95% variance: {n_95} components")
        print(f"  Elbow point: {elbow} components")
        print(f"  Parallel analysis: {n_parallel} components")
        print(f"  Selected: {self.n_components} components")
        
        # Refit PCA with optimal components
        self.fit_pca(n_components=self.n_components)
        
        return self.n_components
    
    def analyze_loadings(self, feature_names):
        """Analyze component loadings to interpret and name PCs."""
        print(f"\n=== ANALYZING LOADINGS ===")
        
        # Get loadings (components)
        self.loadings = self.pca.components_.T
        
        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            self.loadings,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=feature_names
        )
        
        print("Top loadings for each component:")
        for i in range(self.n_components):
            pc_name = f'PC{i+1}'
            top_loadings = loadings_df[pc_name].abs().nlargest(5)
            print(f"\n{pc_name} (Explained Variance: {self.explained_variance_ratio[i]:.3f}):")
            for feature, loading in top_loadings.items():
                direction = "positive" if loadings_df.loc[feature, pc_name] > 0 else "negative"
                print(f"  {feature}: {loadings_df.loc[feature, pc_name]:.3f} ({direction})")
        
        # Name components based on loadings
        component_names = self._name_components(loadings_df)
        
        return loadings_df, component_names
    
    def _name_components(self, loadings_df):
        """Name components based on their loadings."""
        component_names = {}
        
        for i in range(self.n_components):
            pc_name = f'PC{i+1}'
            top_features = loadings_df[pc_name].abs().nlargest(3).index.tolist()
            
            # Analyze the nature of top features
            service_features = [f for f in top_features if any(word in f.lower() 
                             for word in ['service', 'comfort', 'food', 'seat', 'entertainment', 'cleanliness'])]
            convenience_features = [f for f in top_features if any(word in f.lower() 
                                for word in ['delay', 'booking', 'boarding', 'checkin', 'gate'])]
            customer_features = [f for f in top_features if any(word in f.lower() 
                               for word in ['customer', 'gender', 'age', 'travel', 'class'])]
            
            if service_features:
                component_names[pc_name] = "Overall Service Quality"
            elif convenience_features:
                component_names[pc_name] = "Convenience & Efficiency"
            elif customer_features:
                component_names[pc_name] = "Customer Demographics & Travel Type"
            else:
                component_names[pc_name] = f"General Factor {i+1}"
        
        return component_names
    
    def create_loadings_heatmap(self, loadings_df, component_names):
        """Create loadings heatmap with component names."""
        print("\n=== CREATING LOADINGS HEATMAP ===")
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Use absolute values for better visualization
        abs_loadings = loadings_df.abs()
        
        # Create heatmap
        sns.heatmap(abs_loadings, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        # Update x-axis labels with component names
        x_labels = [f'{col}\n({component_names.get(col, "Unknown")})' 
                   for col in loadings_df.columns]
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
        
        plt.title('PCA Loadings Heatmap\n(Feature Contributions to Principal Components)', 
                 fontsize=14, pad=20)
        plt.xlabel('Principal Components')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('loadings_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return abs_loadings
    
    def validate_pca(self):
        """Validate PCA with reconstruction error and stability analysis."""
        print("\n=== VALIDATING PCA ===")
        
        # 1. Reconstruction error analysis
        print("1. Reconstruction Error Analysis:")
        reconstruction_errors = []
        component_range = range(1, min(20, self.X_train.shape[1]) + 1)
        
        for n_comp in component_range:
            pca_temp = PCA(n_components=n_comp)
            pca_temp.fit(self.X_train)
            
            # Transform and inverse transform
            X_train_transformed = pca_temp.transform(self.X_train)
            X_train_reconstructed = pca_temp.inverse_transform(X_train_transformed)
            
            # Calculate MSE
            mse = mean_squared_error(self.X_train, X_train_reconstructed)
            reconstruction_errors.append(mse)
        
        # Plot reconstruction error
        plt.figure(figsize=(10, 6))
        plt.plot(component_range, reconstruction_errors, 'bo-', linewidth=2)
        plt.axvline(x=self.n_components, color='red', linestyle='--', 
                   label=f'Selected: {self.n_components} components')
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction MSE')
        plt.title('PCA Reconstruction Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reconstruction_error.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   Reconstruction MSE with {self.n_components} components: {reconstruction_errors[self.n_components-1]:.4f}")
        
        # 2. Stability analysis (bootstrap)
        print("\n2. Stability Analysis (Bootstrap):")
        n_bootstrap = 50
        bootstrap_loadings = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(self.X_train.shape[0], size=self.X_train.shape[0], replace=True)
            X_bootstrap = self.X_train[indices]
            
            # Fit PCA
            pca_bootstrap = PCA(n_components=self.n_components)
            pca_bootstrap.fit(X_bootstrap)
            bootstrap_loadings.append(pca_bootstrap.components_)
        
        # Calculate loading stability
        bootstrap_loadings = np.array(bootstrap_loadings)
        loading_std = np.std(bootstrap_loadings, axis=0)
        loading_mean = np.mean(bootstrap_loadings, axis=0)
        
        stability_scores = np.mean(loading_std / (np.abs(loading_mean) + 1e-8), axis=0)
        print(f"   Average stability score: {np.mean(stability_scores):.3f}")
        print(f"   Stability by component: {stability_scores}")
        
        # 3. External validation with satisfaction
        print("\n3. External Validation with Satisfaction:")
        X_train_pca = self.pca.transform(self.X_train)
        
        # Convert satisfaction to numeric for correlation
        satisfaction_numeric = (self.y_train == 'satisfied').astype(int)
        
        correlations = []
        for i in range(self.n_components):
            corr = np.corrcoef(X_train_pca[:, i], satisfaction_numeric)[0, 1]
            correlations.append(corr)
            print(f"   PC{i+1} vs Satisfaction: {corr:.3f}")
        
        return reconstruction_errors, stability_scores, correlations
    
    def create_biplot(self, feature_names, component_names):
        """Create biplot showing PC1 vs PC2 with loading vectors."""
        print("\n=== CREATING BIPLOT ===")
        
        # Transform data
        X_train_pca = self.pca.transform(self.X_train)
        
        # Create biplot
        plt.figure(figsize=(12, 10))
        
        # Plot data points
        scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                            alpha=0.6, s=20, c='lightblue', edgecolors='black', linewidth=0.5)
        
        # Plot loading vectors
        scale_factor = 3  # Adjust for visibility
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, 
                     self.loadings[i, 0] * scale_factor, 
                     self.loadings[i, 1] * scale_factor,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
            plt.text(self.loadings[i, 0] * scale_factor * 1.1, 
                    self.loadings[i, 1] * scale_factor * 1.1,
                    feature, fontsize=8, ha='center', va='center')
        
        plt.xlabel(f'PC1 ({component_names.get("PC1", "Unknown")}) - {self.explained_variance_ratio[0]:.1%} variance')
        plt.ylabel(f'PC2 ({component_names.get("PC2", "Unknown")}) - {self.explained_variance_ratio[1]:.1%} variance')
        plt.title('PCA Biplot: PC1 vs PC2\n(Red arrows show feature loadings)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_outliers(self):
        """Check for outliers using Mahalanobis distance in PC space."""
        print("\n=== OUTLIER DETECTION ===")
        
        # Transform data to PC space
        X_train_pca = self.pca.transform(self.X_train)
        
        # Calculate Mahalanobis distance
        mean_pc = np.mean(X_train_pca, axis=0)
        cov_pc = np.cov(X_train_pca.T)
        
        # Calculate Mahalanobis distance for each point
        mahal_distances = []
        for i in range(len(X_train_pca)):
            try:
                inv_cov = np.linalg.inv(cov_pc)
                mahal_dist = mahalanobis(X_train_pca[i], mean_pc, inv_cov)
                mahal_distances.append(mahal_dist)
            except np.linalg.LinAlgError:
                # If covariance matrix is singular, use Euclidean distance
                mahal_dist = np.linalg.norm(X_train_pca[i] - mean_pc)
                mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        
        # Identify outliers (using 95th percentile as threshold)
        threshold = np.percentile(mahal_distances, 95)
        outliers = mahal_distances > threshold
        
        print(f"   Mahalanobis distance threshold (95th percentile): {threshold:.3f}")
        print(f"   Number of outliers detected: {np.sum(outliers)}")
        print(f"   Percentage of outliers: {np.sum(outliers) / len(outliers) * 100:.2f}%")
        
        # Plot outlier detection
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(mahal_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.xlabel('Mahalanobis Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Mahalanobis Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        X_train_pca_2d = X_train_pca[:, :2]
        plt.scatter(X_train_pca_2d[~outliers, 0], X_train_pca_2d[~outliers, 1], 
                   alpha=0.6, s=20, c='lightblue', label='Normal points')
        plt.scatter(X_train_pca_2d[outliers, 0], X_train_pca_2d[outliers, 1], 
                   alpha=0.8, s=30, c='red', label='Outliers')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Outliers in PC Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outliers, mahal_distances
    
    def create_feature_contributions_plot(self, loadings_df, component_names):
        """Create bar chart of absolute loadings for top components."""
        print("\n=== CREATING FEATURE CONTRIBUTIONS PLOT ===")
        
        # Select top 2 components for detailed analysis
        n_components_to_plot = min(2, self.n_components)
        
        fig, axes = plt.subplots(1, n_components_to_plot, figsize=(15, 6))
        if n_components_to_plot == 1:
            axes = [axes]
        
        for i in range(n_components_to_plot):
            pc_name = f'PC{i+1}'
            component_name = component_names.get(pc_name, f'Component {i+1}')
            
            # Get top 10 features by absolute loading
            top_features = loadings_df[pc_name].abs().nlargest(10)
            
            # Create bar plot
            bars = axes[i].bar(range(len(top_features)), top_features.values, 
                              color='skyblue', edgecolor='black', alpha=0.7)
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Absolute Loading')
            axes[i].set_title(f'{pc_name}: {component_name}\n({self.explained_variance_ratio[i]:.1%} variance)')
            axes[i].set_xticks(range(len(top_features)))
            axes[i].set_xticklabels(top_features.index, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('feature_contributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, train_path, test_path):
        """Run the complete Phase 2 PCA analysis."""
        print("="*60)
        print("PHASE 2 - PCA ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_processed_data(train_path, test_path)
        
        # Get feature names
        feature_names = [col for col in self.train_df.columns if col != 'satisfaction']
        
        # Determine optimal components
        self.determine_optimal_components()
        
        # Analyze loadings and name components
        loadings_df, component_names = self.analyze_loadings(feature_names)
        
        # Create visualizations
        self.create_loadings_heatmap(loadings_df, component_names)
        self.create_biplot(feature_names, component_names)
        self.create_feature_contributions_plot(loadings_df, component_names)
        
        # Validate PCA
        reconstruction_errors, stability_scores, correlations = self.validate_pca()
        
        # Check for outliers
        outliers, mahal_distances = self.check_outliers()
        
        # Generate summary report
        self._generate_summary_report(component_names, reconstruction_errors, 
                                    stability_scores, correlations, outliers)
        
        print("\n" + "="*60)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self
    
    def _generate_summary_report(self, component_names, reconstruction_errors, 
                               stability_scores, correlations, outliers):
        """Generate comprehensive summary report."""
        print("\n=== PCA ANALYSIS SUMMARY ===")
        
        print(f"Selected Components: {self.n_components}")
        print(f"Total Explained Variance: {self.cumulative_variance[self.n_components-1]:.3f}")
        
        print("\nComponent Interpretations:")
        for i in range(self.n_components):
            pc_name = f'PC{i+1}'
            print(f"  {pc_name}: {component_names.get(pc_name, 'Unknown')} "
                  f"({self.explained_variance_ratio[i]:.1%} variance)")
        
        print(f"\nValidation Results:")
        print(f"  Reconstruction MSE: {reconstruction_errors[self.n_components-1]:.4f}")
        print(f"  Average Stability Score: {np.mean(stability_scores):.3f}")
        print(f"  Outliers Detected: {np.sum(outliers)} ({np.sum(outliers)/len(outliers)*100:.1f}%)")
        
        print(f"\nExternal Validation (PC-Satisfaction Correlations):")
        for i, corr in enumerate(correlations):
            print(f"  PC{i+1}: {corr:.3f}")

def main():
    """Main function to run the PCA analysis."""
    # Initialize PCA analysis
    pca_analysis = PCAAnalysis()
    
    # Run complete analysis
    pca_analysis.run_complete_analysis(
        train_path='train_processed.csv',
        test_path='test_processed.csv'
    )

if __name__ == "__main__":
    main()
