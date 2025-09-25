"""
PCA utilities for airline passenger satisfaction analysis.

This module provides reusable functions for Principal Component Analysis
and dimensionality reduction used in the airline passenger segmentation project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def fit_pca(X, n_components=None, random_state=42):
    """
    Fit PCA on the given data.
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_components : int, optional
        Number of components. If None, use all components
    random_state : int, default 42
        Random state for reproducibility
        
    Returns:
    --------
    sklearn.decomposition.PCA
        Fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    
    print(f"PCA fitted with {pca.n_components_} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return pca


def determine_optimal_components(X, max_components=None, variance_threshold=0.8):
    """
    Determine optimal number of components using multiple criteria.
    
    Parameters:
    -----------
    X : array-like
        Input data
    max_components : int, optional
        Maximum number of components to test. If None, use min(n_features, 20)
    variance_threshold : float, default 0.8
        Minimum cumulative variance to retain
        
    Returns:
    --------
    dict
        Dictionary with optimal K and analysis results
    """
    if max_components is None:
        max_components = min(X.shape[1], 20)
    
    # Fit PCA with all components first
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find components for variance threshold
    n_variance = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    
    # Find elbow point (simplified)
    diffs = np.diff(explained_variance)
    diffs2 = np.diff(diffs)
    elbow_point = np.argmax(diffs2) + 2
    
    # Choose conservative approach
    optimal_k = min(n_variance, elbow_point)
    
    results = {
        'optimal_k': optimal_k,
        'n_variance': n_variance,
        'elbow_point': elbow_point,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance
    }
    
    print(f"Optimal components: {optimal_k}")
    print(f"Components for {variance_threshold*100}% variance: {n_variance}")
    print(f"Elbow point: {elbow_point}")
    
    return results


def create_scree_plot(pca_results, save_path=None):
    """
    Create scree plot and cumulative variance plot.
    
    Parameters:
    -----------
    pca_results : dict
        Results from determine_optimal_components
    save_path : str, optional
        Path to save the plot
    """
    explained_variance = pca_results['explained_variance']
    cumulative_variance = pca_results['cumulative_variance']
    optimal_k = pca_results['optimal_k']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scree plot
    components = range(1, len(explained_variance) + 1)
    ax1.plot(components, explained_variance, 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
               label=f'Optimal K={optimal_k}')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(components, cumulative_variance, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
               label=f'Optimal K={optimal_k}')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_loadings(pca, feature_names, n_top=5):
    """
    Analyze PCA loadings and identify important features for each component.
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    feature_names : list
        List of feature names
    n_top : int, default 5
        Number of top features to show for each component
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with loadings for each component
    """
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    
    print("Top loadings for each component:")
    for i in range(pca.n_components_):
        pc_name = f'PC{i+1}'
        top_loadings = loadings_df[pc_name].abs().nlargest(n_top)
        print(f"\n{pc_name} (Explained Variance: {pca.explained_variance_ratio_[i]:.3f}):")
        for feature, loading in top_loadings.items():
            direction = "positive" if loadings_df.loc[feature, pc_name] > 0 else "negative"
            print(f"  {feature}: {loadings_df.loc[feature, pc_name]:.3f} ({direction})")
    
    return loadings_df


def create_loadings_heatmap(loadings_df, save_path=None):
    """
    Create heatmap of PCA loadings.
    
    Parameters:
    -----------
    loadings_df : pandas.DataFrame
        DataFrame with loadings
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Use absolute values for better visualization
    abs_loadings = loadings_df.abs()
    
    sns.heatmap(abs_loadings, 
               annot=True, 
               cmap='RdYlBu_r', 
               center=0,
               fmt='.2f',
               cbar_kws={"shrink": .8})
    
    plt.title('PCA Loadings Heatmap', fontsize=14, pad=20)
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def validate_pca(pca, X, n_components_range=None):
    """
    Validate PCA using reconstruction error.
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    X : array-like
        Original data
    n_components_range : range, optional
        Range of components to test. If None, use 1 to n_components
        
    Returns:
    --------
    dict
        Validation results including reconstruction errors
    """
    if n_components_range is None:
        n_components_range = range(1, pca.n_components_ + 1)
    
    reconstruction_errors = []
    
    for n_comp in n_components_range:
        pca_temp = PCA(n_components=n_comp)
        pca_temp.fit(X)
        
        # Transform and inverse transform
        X_transformed = pca_temp.transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)
        
        # Calculate MSE
        mse = mean_squared_error(X, X_reconstructed)
        reconstruction_errors.append(mse)
    
    results = {
        'n_components': list(n_components_range),
        'reconstruction_errors': reconstruction_errors
    }
    
    return results


def create_biplot(X_pca, loadings_df, feature_names, n_features=10, save_path=None):
    """
    Create biplot showing data points and feature loadings.
    
    Parameters:
    -----------
    X_pca : array-like
        PCA-transformed data (first 2 components)
    loadings_df : pandas.DataFrame
        DataFrame with loadings
    feature_names : list
        List of feature names
    n_features : int, default 10
        Number of top features to show
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20, c='lightblue', edgecolors='black', linewidth=0.5)
    
    # Plot loading vectors for top features
    scale_factor = 3
    top_features = loadings_df.abs().sum(axis=1).nlargest(n_features).index
    
    for feature in top_features:
        plt.arrow(0, 0, 
                 loadings_df.loc[feature, 'PC1'] * scale_factor, 
                 loadings_df.loc[feature, 'PC2'] * scale_factor,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        plt.text(loadings_df.loc[feature, 'PC1'] * scale_factor * 1.1, 
                loadings_df.loc[feature, 'PC2'] * scale_factor * 1.1,
                feature, fontsize=8, ha='center', va='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def pca_pipeline(X, feature_names, variance_threshold=0.8, save_plots=True):
    """
    Complete PCA pipeline.
    
    Parameters:
    -----------
    X : array-like
        Input data
    feature_names : list
        List of feature names
    variance_threshold : float, default 0.8
        Minimum cumulative variance to retain
    save_plots : bool, default True
        Whether to save plots
        
    Returns:
    --------
    dict
        Complete PCA results including model, loadings, and analysis
    """
    print("Starting PCA pipeline...")
    
    # Determine optimal components
    pca_results = determine_optimal_components(X, variance_threshold=variance_threshold)
    optimal_k = pca_results['optimal_k']
    
    # Fit final PCA
    pca = fit_pca(X, n_components=optimal_k)
    
    # Transform data
    X_pca = pca.transform(X)
    
    # Analyze loadings
    loadings_df = analyze_loadings(pca, feature_names)
    
    # Create visualizations
    if save_plots:
        create_scree_plot(pca_results, 'scree_plot.png')
        create_loadings_heatmap(loadings_df, 'loadings_heatmap.png')
        create_biplot(X_pca[:, :2], loadings_df, feature_names, save_path='pca_biplot.png')
    
    # Validate PCA
    validation_results = validate_pca(pca, X)
    
    results = {
        'pca': pca,
        'X_pca': X_pca,
        'loadings_df': loadings_df,
        'pca_results': pca_results,
        'validation_results': validation_results
    }
    
    print("PCA pipeline completed successfully!")
    return results


if __name__ == "__main__":
    # Example usage
    from preprocess import preprocess_pipeline
    
    # Load and preprocess data
    train_df, test_df, feature_names = preprocess_pipeline(
        '../archive-2/train.csv',
        '../archive-2/test.csv'
    )
    
    # Prepare features
    X_train = train_df[feature_names].values
    
    # Run PCA pipeline
    pca_results = pca_pipeline(X_train, feature_names)
