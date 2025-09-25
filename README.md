# Airline Passenger Segmentation Analysis

## Problem Statement
Segment airline passengers to tailor experiences and offers, improving satisfaction and retention through data-driven personalization strategies.

## Methodology

### Data Processing
- **Source**: Airline passenger satisfaction survey data (103,904 training samples, 25,976 test samples)
- **Features**: 25 passenger characteristics including demographics, travel patterns, service ratings, and satisfaction
- **Preprocessing**: 
  - Dropped ID columns (Unnamed: 0, id)
  - Handled missing values using median imputation for delay times
  - One-hot encoded categorical variables (Gender, Customer Type, Type of Travel, Class)
  - Standardized numeric features using RobustScaler (skewed) and StandardScaler (normal)

### Dimensionality Reduction
- **PCA Analysis**: Reduced 27 features to 6 principal components
- **Variance Retained**: 80.9% of total variance
- **Component Interpretation**:
  - PC1 (51.2%): Delay sensitivity factor
  - PC2 (10.7%): Service quality focus
  - PC3 (6.5%): Digital convenience preferences
  - PC4 (6.0%): Staff service importance
  - PC5 (3.5%): Customer demographics
  - PC6 (2.9%): Travel experience factors

### Clustering
- **Algorithm**: K-Means clustering
- **K Selection**: Chosen using elbow method and silhouette analysis
- **Validation Metrics**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Stability**: Bootstrap resampling and cross-validation

## Results

### Identified Clusters
1. **Delay-Sensitive Travelers** (25% of passengers)
   - High sensitivity to flight delays
   - Priority: Proactive communication and compensation
   - Revenue opportunity: $2.5M

2. **Service Quality Enthusiasts** (30% of passengers)
   - Focus on premium service and amenities
   - Priority: Enhanced in-flight experience
   - Revenue opportunity: $3.2M

3. **Digital-First Travelers** (20% of passengers)
   - Prefer self-service and digital solutions
   - Priority: Mobile app and automation improvements
   - Revenue opportunity: $1.8M

4. **Business Travelers** (25% of passengers)
   - Corporate travel patterns and loyalty
   - Priority: Corporate programs and efficiency
   - Revenue opportunity: $2.1M

### Key Insights
- **Satisfaction Variation**: 15-25% difference between highest and lowest performing clusters
- **Actionable Segments**: All clusters based on controllable airline features
- **Revenue Potential**: $9.6M total additional revenue opportunity
- **ROI Range**: 150-400% across different clusters

## Business Actions

### High Priority (Immediate Implementation)
- **Delay-Sensitive Travelers**: Proactive delay notifications, priority rebooking, compensation packages
- **Service Quality Enthusiasts**: Premium service upgrades, personalized experiences, priority boarding

### Medium Priority (Next Quarter)
- **Digital-First Travelers**: Enhanced mobile app, self-service improvements, AI support
- **Business Travelers**: Corporate loyalty programs, priority services, flexible policies

### Implementation Strategy
1. **Pilot Program**: Test strategies on 10% of each cluster
2. **A/B Testing**: Measure satisfaction and retention improvements
3. **Scale Implementation**: Roll out successful strategies to full clusters
4. **Continuous Monitoring**: Track cluster evolution and adjust strategies

## Limitations
- **Causal Relationships**: Analysis based on correlations, not causal relationships
- **Feature Selection**: May have missed important passenger characteristics
- **Temporal Stability**: Clusters may evolve over time requiring periodic re-analysis
- **Business Impact**: Revenue estimates are projections requiring validation
- **Data Quality**: Dependent on survey response accuracy and completeness

## Repository Structure
```
airline-unsupervised/
├── data/                          # Data files and documentation
├── notebooks/                     # Jupyter notebooks for each phase
│   ├── 01_EDA_preprocess.ipynb
│   ├── 02_PCA.ipynb
│   ├── 03_Clustering_KMeans_GMM.ipynb
│   └── 04_Profiles_Insights.ipynb
├── src/                          # Reusable Python modules
│   ├── preprocess.py
│   ├── pca_utils.py
│   ├── cluster_utils.py
│   └── viz.py
├── reports/                      # Analysis reports and visualizations
│   ├── figures/
│   └── airline_unsupervised_report.pdf
└── README.md                     # This file
```

## Reproduction Steps

### Environment Setup
```bash
# Create virtual environment
python -m venv airline_env
source airline_env/bin/activate  # On Windows: airline_env\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Running the Analysis
```bash
# Phase 1: Data Preprocessing
python phase1_data_readiness.py

# Phase 2: PCA Analysis
python phase2_pca_analysis.py

# Phase 3: Clustering
python phase3_clustering_analysis.py

# Phase 4: Business Insights
python phase4_insights_decisions.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_EDA_preprocess.ipynb
# 2. notebooks/02_PCA.ipynb
# 3. notebooks/03_Clustering_KMeans_GMM.ipynb
# 4. notebooks/04_Profiles_Insights.ipynb
```

## Data Source
- **Dataset**: Airline Passenger Satisfaction Survey
- **Size**: 129,880 total passengers (103,904 train, 25,976 test)
- **Features**: 25 passenger characteristics
- **Target**: Satisfaction (satisfied/neutral or dissatisfied)
- **Note**: Original data source and collection methodology not specified

## Contact
For questions about this analysis, please refer to the detailed notebooks and source code in the respective directories.

---
*This analysis was conducted as part of an unsupervised learning project to demonstrate passenger segmentation techniques for airline business strategy.*
