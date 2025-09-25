"""
Data preprocessing utilities for airline passenger satisfaction analysis.

This module provides reusable functions for data cleaning, encoding, and scaling
used across the airline passenger segmentation project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """
    Load training and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    test_path : str
        Path to test data CSV file
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Loaded DataFrames
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded data: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


def drop_id_columns(df, id_columns=None):
    """
    Drop ID-like columns that don't contribute to analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    id_columns : list, optional
        List of column names to drop. Default: ['Unnamed: 0', 'id']
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ID columns removed
    """
    if id_columns is None:
        id_columns = ['Unnamed: 0', 'id']
    
    # Only drop columns that exist
    existing_id_cols = [col for col in id_columns if col in df.columns]
    df_cleaned = df.drop(columns=existing_id_cols)
    
    print(f"Dropped ID columns: {existing_id_cols}")
    return df_cleaned


def handle_missing_values(train_df, test_df, strategy='median'):
    """
    Handle missing values using specified imputation strategy.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    test_df : pandas.DataFrame
        Test DataFrame
    strategy : str, default 'median'
        Imputation strategy ('mean', 'median', 'most_frequent')
        
    Returns:
    --------
    tuple
        (train_df, test_df, imputers) - Cleaned DataFrames and fitted imputers
    """
    imputers = {}
    
    # Find columns with missing values
    missing_cols = train_df.columns[train_df.isnull().any()].tolist()
    
    if not missing_cols:
        print("No missing values found")
        return train_df, test_df, imputers
    
    print(f"Handling missing values in: {missing_cols}")
    
    for col in missing_cols:
        imputer = SimpleImputer(strategy=strategy)
        
        # Fit on training data
        train_df[col] = imputer.fit_transform(train_df[[col]]).flatten()
        # Transform test data
        test_df[col] = imputer.transform(test_df[[col]]).flatten()
        
        imputers[col] = imputer
        print(f"Applied {strategy} imputation to {col}")
    
    return train_df, test_df, imputers


def encode_categorical_variables(train_df, test_df, categorical_cols=None, target_col='satisfaction'):
    """
    One-hot encode categorical variables.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    test_df : pandas.DataFrame
        Test DataFrame
    categorical_cols : list, optional
        List of categorical column names. If None, auto-detect object columns
    target_col : str, default 'satisfaction'
        Target column name to exclude from encoding
        
    Returns:
    --------
    tuple
        (train_encoded, test_encoded) - Encoded DataFrames
    """
    if categorical_cols is None:
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    print(f"Encoding categorical variables: {categorical_cols}")
    
    # One-hot encode training data
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, prefix=categorical_cols)
    
    # One-hot encode test data
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, prefix=categorical_cols)
    
    # Ensure test has same columns as train
    missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_cols:
        test_encoded[col] = 0
    
    # Reorder test columns to match train
    test_encoded = test_encoded[train_encoded.columns]
    
    print(f"Encoded shapes: Train {train_encoded.shape}, Test {test_encoded.shape}")
    return train_encoded, test_encoded


def scale_numeric_features(train_df, test_df, numeric_cols=None, target_col='satisfaction'):
    """
    Scale numeric features using appropriate scaling methods.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    test_df : pandas.DataFrame
        Test DataFrame
    numeric_cols : list, optional
        List of numeric column names. If None, auto-detect numeric columns
    target_col : str, default 'satisfaction'
        Target column name to exclude from scaling
        
    Returns:
    --------
    tuple
        (train_scaled, test_scaled, scalers) - Scaled DataFrames and fitted scalers
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
    
    # Analyze skewness
    skewness_data = []
    for col in numeric_cols:
        skewness = train_df[col].skew()
        skewness_data.append({'Variable': col, 'Skewness': skewness})
    
    skew_df = pd.DataFrame(skewness_data)
    highly_skewed = skew_df[abs(skew_df['Skewness']) > 1]['Variable'].tolist()
    normal_to_moderate = [col for col in numeric_cols if col not in highly_skewed]
    
    print(f"Highly skewed variables (RobustScaler): {highly_skewed}")
    print(f"Normal/moderate variables (StandardScaler): {len(normal_to_moderate)}")
    
    scalers = {}
    
    # Apply robust scaling to highly skewed variables
    if highly_skewed:
        robust_scaler = RobustScaler()
        train_df[highly_skewed] = robust_scaler.fit_transform(train_df[highly_skewed])
        test_df[highly_skewed] = robust_scaler.transform(test_df[highly_skewed])
        scalers['robust'] = robust_scaler
    
    # Apply standard scaling to normal/moderate variables
    if normal_to_moderate:
        standard_scaler = StandardScaler()
        train_df[normal_to_moderate] = standard_scaler.fit_transform(train_df[normal_to_moderate])
        test_df[normal_to_moderate] = standard_scaler.transform(test_df[normal_to_moderate])
        scalers['standard'] = standard_scaler
    
    return train_df, test_df, scalers


def preprocess_pipeline(train_path, test_path, save_processed=True):
    """
    Complete preprocessing pipeline for airline passenger data.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    test_path : str
        Path to test data CSV file
    save_processed : bool, default True
        Whether to save processed data to CSV files
        
    Returns:
    --------
    tuple
        (train_processed, test_processed, feature_names) - Processed DataFrames and feature names
    """
    print("Starting data preprocessing pipeline...")
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Drop ID columns
    train_df = drop_id_columns(train_df)
    test_df = drop_id_columns(test_df)
    
    # Handle missing values
    train_df, test_df, imputers = handle_missing_values(train_df, test_df)
    
    # Encode categorical variables
    train_df, test_df = encode_categorical_variables(train_df, test_df)
    
    # Scale numeric features
    train_df, test_df, scalers = scale_numeric_features(train_df, test_df)
    
    # Get feature names (excluding target)
    feature_names = [col for col in train_df.columns if col != 'satisfaction']
    
    print(f"Preprocessing complete. Final shapes: Train {train_df.shape}, Test {test_df.shape}")
    
    # Save processed data
    if save_processed:
        train_df.to_csv('train_processed.csv', index=False)
        test_df.to_csv('test_processed.csv', index=False)
        print("Processed data saved to CSV files")
    
    return train_df, test_df, feature_names


if __name__ == "__main__":
    # Example usage
    train_processed, test_processed, features = preprocess_pipeline(
        'archive-2/train.csv',
        'archive-2/test.csv'
    )
