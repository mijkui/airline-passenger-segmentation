#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataReadinessPipeline:
    """Complete data readiness pipeline for airline passenger satisfaction data."""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.feature_info = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self, train_path, test_path):
        """Load train and test datasets."""
        print("Loading datasets...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        return self
    
    def drop_id_columns(self):
        """Drop ID-like columns that don't contribute to analysis."""
        print("\nDROPPING ID COLUMNS Dropping ID columns")
        
        # Identify ID-like columns
        id_columns = ['Unnamed: 0', 'id']
        
        print(f"Dropping columns: {id_columns}")
        
        # Drop from both datasets
        self.train_df = self.train_df.drop(columns=id_columns)
        self.test_df = self.test_df.drop(columns=id_columns)
        
        print(f"Train shape after dropping IDs: {self.train_df.shape}")
        print(f"Test shape after dropping IDs: {self.test_df.shape}")
        
        return self
    
    def analyze_missing_values(self):
        """Analyze missing values before and after imputation."""
        print("\nMISSING VALUES ANALYSIS Missing values analysis")
        
        # Before imputation
        print("Missing values BEFORE imputation:")
        missing_before = self.train_df.isnull().sum() 
        missing_pct_before = (missing_before / len(self.train_df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_before,
            'Missing_Percentage': missing_pct_before
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        return missing_before
    
    def handle_missing_values(self):
        """Handle missing values with appropriate imputation strategies."""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Identify columns with missing values
        missing_cols = self.train_df.columns[self.train_df.isnull().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")
        
        # Document imputation strategy
        imputation_strategy = {
            'Arrival Delay in Minutes': 'median'  # Use median for delay times (robust to outliers)
        }
        
        print("Imputation strategy:")
        for col, strategy in imputation_strategy.items():
            print(f"  {col}: {strategy}")
        
        # Apply imputation
        for col in missing_cols:
            if col in imputation_strategy:
                strategy = imputation_strategy[col]
                imputer = SimpleImputer(strategy=strategy) # make 'Arrival Delay in Minutes' use median to impute missing values
                
                # Fit on train data
                self.train_df[col] = imputer.fit_transform(self.train_df[[col]]).flatten()
                # Transform test data
                self.test_df[col] = imputer.transform(self.test_df[[col]]).flatten()
                
                # Store imputer for later use
                self.imputers[col] = imputer
                
                print(f"Applied {strategy} imputation to {col}")
        
        # Verify no missing values remain
        print(f"Missing values after imputation: {self.train_df.isnull().sum().sum()}")
        
        return self
    
    def identify_categorical_variables(self):
        """Identify categorical variables for encoding."""
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        # Remove target variable from categorical encoding
        if 'satisfaction' in categorical_cols:
            categorical_cols.remove('satisfaction')
        
        print(f"\nCategorical variables to encode: {categorical_cols}")
        
        # Document unique values for each caategorical varible
        for col in categorical_cols:
            unique_vals = self.train_df[col].unique()
            print(f"  {col}: {unique_vals}")
        
        return categorical_cols
    
    def one_hot_encode_categorical(self, categorical_cols):
        """One-hot encode categorical variables."""
        print("\nOne-hot encoding categorical variables")
        
        # Store original categorical columns for reference
        self.categorical_columns = categorical_cols.copy()
        
        # One-hot encode train data
        train_encoded = pd.get_dummies(self.train_df, columns=categorical_cols, prefix=categorical_cols)
        
        # One-hot encode test data (ensure same columns as train)
        test_encoded = pd.get_dummies(self.test_df, columns=categorical_cols, prefix=categorical_cols)
        
        # Ensure test has same columns as train (add missing columns with 0s)
        missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
        for col in missing_cols:
            test_encoded[col] = 0
        
        # Reorder test columns to match train
        test_encoded = test_encoded[train_encoded.columns]
        
        self.train_df = train_encoded
        self.test_df = test_encoded
        
        print(f"Train shape after encoding: {self.train_df.shape}")
        print(f"Test shape after encoding: {self.test_df.shape}")
        print(f"New columns: {[col for col in self.train_df.columns if any(cat in col for cat in categorical_cols)]}")
        
        return self
    
    def identify_numeric_variables(self):
        """Identify numeric variables for scaling."""
        # Exclude target variable
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'satisfaction' in numeric_cols:
            numeric_cols.remove('satisfaction')
        
        print(f"\nNumeric variables to scale: {numeric_cols}")
        return numeric_cols
    
    def analyze_skewness(self, numeric_cols):
        """Analyze skewness of numeric variables."""
        print("\nSkewness Analysis\n")
        
        skewness_data = []
        for col in numeric_cols:
            skewness = self.train_df[col].skew()
            skewness_data.append({
                'Variable': col,
                'Skewness': skewness,
                'Skew_Level': 'Highly Skewed' if abs(skewness) > 1 else 'Moderately Skewed' if abs(skewness) > 0.5 else 'Approximately Normal'
            })
        
        skew_df = pd.DataFrame(skewness_data).sort_values('Skewness', key=abs, ascending=False)
        print(skew_df)
        
        return skew_df
    
    def standardize_numeric_features(self, numeric_cols, skewness_df):
        """Standardize numeric features using appropriate scaling methods."""
        print("\nStandardizeing numeric features")
        
        # Separate highly skewed variables for robust scaling
        highly_skewed = skewness_df[skewness_df['Skew_Level'] == 'Highly Skewed']['Variable'].tolist()
        normal_to_moderate = [col for col in numeric_cols if col not in highly_skewed]
        
        print(f"Highly skewed variables (using RobustScaler): {highly_skewed}")
        print(f"Normal to moderate variables (using StandardScaler): {normal_to_moderate}")
        
        # Apply robust scaling to highly skewed variables
        if highly_skewed:
            robust_scaler = RobustScaler()
            self.train_df[highly_skewed] = robust_scaler.fit_transform(self.train_df[highly_skewed])
            self.test_df[highly_skewed] = robust_scaler.transform(self.test_df[highly_skewed])
            self.scalers['robust'] = robust_scaler
            print(f"Applied RobustScaler to: {highly_skewed}")
        
        # Apply standard scaling to normal/moderate variables
        if normal_to_moderate:
            standard_scaler = StandardScaler()
            self.train_df[normal_to_moderate] = standard_scaler.fit_transform(self.train_df[normal_to_moderate])
            self.test_df[normal_to_moderate] = standard_scaler.transform(self.test_df[normal_to_moderate])
            self.scalers['standard'] = standard_scaler
            print(f"Applied StandardScaler to: {normal_to_moderate}")
        
        return self
    
    def create_feature_summary_table(self, categorical_cols, numeric_cols, skewness_df):
        """Create comprehensive feature summary table."""
        print("\nCREATING FEATURE SUMMARY TABLE Creating feature summary table")
        
        feature_info = []
        
        # Add categorical features
        for col in categorical_cols:
            # Get the encoded column names for this categorical variable
            encoded_cols = [c for c in self.train_df.columns if c.startswith(f"{col}_")]
            # Extract the unique values from the column names
            unique_values = [c.split('_', 1)[1] for c in encoded_cols]
            
            feature_info.append({
                'Feature': col,
                'Type': 'Categorical',
                'Transform_Applied': 'One-hot encoding',
                'Original_Values': str(unique_values),
                'New_Columns': len(encoded_cols)
            })
        
        # Add numeric features
        for col in numeric_cols:
            skew_level = skewness_df[skewness_df['Variable'] == col]['Skew_Level'].iloc[0] if col in skewness_df['Variable'].values else 'Unknown'
            transform = 'RobustScaler' if skew_level == 'Highly Skewed' else 'StandardScaler'
            
            feature_info.append({
                'Feature': col,
                'Type': 'Numeric',
                'Transform_Applied': transform,
                'Original_Values': f"Range: {self.train_df[col].min():.2f} to {self.train_df[col].max():.2f}",
                'New_Columns': 1
            })
        
        # Add target variable
        feature_info.append({
            'Feature': 'satisfaction',
            'Type': 'Target',
            'Transform_Applied': 'None (kept as original)',
            'Original_Values': str(self.train_df['satisfaction'].unique().tolist()),
            'New_Columns': 1
        })
        
        self.feature_info = pd.DataFrame(feature_info)
        print(self.feature_info.to_string(index=False))
        
        return self.feature_info
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap of numeric features."""
        print("\nCreating correlation heatmap")
        
        # Get numeric columns (excluding target)
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'satisfaction' in numeric_cols:
            numeric_cols.remove('satisfaction')
        
        # Calculate correlation matrix
        corr_matrix = self.train_df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap\n(Shows redundancy → justifies PCA)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"\nHigh correlation pairs (>0.7):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
        else:
            print("\nNo high correlation pairs found (>0.7 threshold)")
        
        return corr_matrix
    
    def create_distribution_plots(self, numeric_cols):
        """Create distribution plots before and after transformation."""
        print("\nCreating distribution plots")
        
        # Load original data for comparison
        original_train = pd.read_csv('archive-2/train.csv')
        
        # Select a few key numeric variables for plotting
        key_vars = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
        key_vars = [var for var in key_vars if var in numeric_cols]
        
        fig, axes = plt.subplots(len(key_vars), 2, figsize=(15, 4*len(key_vars)))
        if len(key_vars) == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(key_vars):
            # Before transformation
            axes[i, 0].hist(original_train[var].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i, 0].set_title(f'{var} - Before Transformation', fontsize=12)
            axes[i, 0].set_xlabel(var)
            axes[i, 0].set_ylabel('Frequency')
            
            # After transformation
            axes[i, 1].hist(self.train_df[var], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[i, 1].set_title(f'{var} - After Transformation', fontsize=12)
            axes[i, 1].set_xlabel(f'{var} (scaled)')
            axes[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_missingness_report(self, missing_before):
        """Generate comprehensive missingness report."""
        print("\nMissing report")
        
        missing_after = self.train_df.isnull().sum()
        
        print("Missing values BEFORE imputation:")
        print(f"Total missing values: {missing_before.sum()}")
        print("Columns with missing values:")
        for col in missing_before.index:
            if missing_before[col] > 0:
                print(f"  {col}: {missing_before[col]} ({missing_before[col]/len(self.train_df)*100:.2f}%)")
        
        print(f"\nMissing values AFTER imputation:")
        print(f"Total missing values: {missing_after.sum()}")
        
        if missing_after.sum() == 0:
            print("All missing values successfully handled!")
        else:
            print(" Some missing values remain:")
            for col in missing_after.index:
                if missing_after[col] > 0:
                    print(f"  {col}: {missing_after[col]}")
        
        return missing_after
    
    def save_processed_data(self):
        """Save processed datasets."""
        print("\nDaving processed data")
        
        self.train_df.to_csv('train_processed.csv', index=False)
        self.test_df.to_csv('test_processed.csv', index=False)
        
        print("Processed data saved:")
        print(f"  - train_processed.csv: {self.train_df.shape}")
        print(f"  - test_processed.csv: {self.test_df.shape}")
        
        return self
    
    def run_complete_pipeline(self, train_path, test_path):
        """Run the complete Phase 1 data readiness pipeline."""
        print("="*60)
        print("PHASE 1 - DATA READINESS PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Analyze missing values before processing
        missing_before = self.analyze_missing_values()
        
        # Drop ID columns
        self.drop_id_columns()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Identify and encode categorical variables
        categorical_cols = self.identify_categorical_variables()
        self.one_hot_encode_categorical(categorical_cols)
        
        # Identify and scale numeric variables
        numeric_cols = self.identify_numeric_variables()
        skewness_df = self.analyze_skewness(numeric_cols)
        self.standardize_numeric_features(numeric_cols, skewness_df)
        
        # Create validation reports
        self.create_feature_summary_table(categorical_cols, numeric_cols, skewness_df)
        self.generate_missingness_report(missing_before)
        
        # Create visualizations
        self.create_correlation_heatmap()
        self.create_distribution_plots(numeric_cols)
        
        # Save processed data
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self

def main():
    """Main function to run the pipeline."""
    # Initialize pipeline
    pipeline = DataReadinessPipeline()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(
        train_path='archive-2/train.csv',
        test_path='archive-2/test.csv'
    )

if __name__ == "__main__":
    main()
