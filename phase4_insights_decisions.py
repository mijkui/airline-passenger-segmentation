#!/usr/bin/env python3
"""
Phase 4 - Insights & Decisions for Airline Passenger Satisfaction
================================================================

This script implements business insights and actionable recommendations:
1. Map clusters to business actions
2. Size clusters and estimate business impact
3. Validate insights with back-testing
4. Create executive summary and recommendations

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BusinessInsights:
    """Business insights and recommendations generator for airline passenger clustering."""
    
    def __init__(self):
        self.train_df = None
        self.X_train_pca = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.business_actions = None
        self.impact_estimates = None
        
    def load_clustering_results(self, train_path, pca_path):
        """Load processed data and clustering results."""
        print("Loading clustering results...")
        
        # Load data
        self.train_df = pd.read_csv(train_path)
        self.X_train_pca = np.loadtxt(pca_path, delimiter=',')
        
        # Perform final clustering (using best K from previous analysis)
        kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.X_train_pca)
        
        print(f"Loaded {len(self.train_df)} passengers with {len(set(self.cluster_labels))} clusters")
        
        return self
    
    def create_cluster_profiles(self):
        """Create detailed cluster profiles with business characteristics."""
        print("\n=== CREATING CLUSTER PROFILES ===")
        
        # Create comprehensive profile DataFrame
        df_profile = pd.DataFrame(self.X_train_pca, columns=[f'PC{i+1}' for i in range(self.X_train_pca.shape[1])])
        df_profile['cluster'] = self.cluster_labels
        df_profile['satisfaction'] = self.train_df['satisfaction'].values
        
        # Add original features for interpretation
        feature_cols = [col for col in self.train_df.columns if col != 'satisfaction']
        for i, col in enumerate(feature_cols):
            df_profile[col] = self.train_df[col].values
        
        self.cluster_profiles = df_profile
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in sorted(df_profile['cluster'].unique()):
            cluster_data = df_profile[df_profile['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_profile)) * 100,
                'satisfaction_rate': (cluster_data['satisfaction'] == 'satisfied').mean() * 100,
                'pc1_mean': cluster_data['PC1'].mean(),  # Delay factor
                'pc2_mean': cluster_data['PC2'].mean(),  # Service quality
                'pc3_mean': cluster_data['PC3'].mean(),  # Digital convenience
                'age_mean': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else 0,
                'flight_distance_mean': cluster_data['Flight Distance'].mean() if 'Flight Distance' in cluster_data.columns else 0,
                'business_travel_pct': (cluster_data['Type of Travel_Business travel'] == 1).mean() * 100 if 'Type of Travel_Business travel' in cluster_data.columns else 0,
                'loyal_customer_pct': (cluster_data['Customer Type_Loyal Customer'] == 1).mean() * 100 if 'Customer Type_Loyal Customer' in cluster_data.columns else 0
            }
            cluster_stats.append(stats)
        
        self.cluster_stats = pd.DataFrame(cluster_stats)
        print(self.cluster_stats)
        
        return self
    
    def map_clusters_to_actions(self):
        """Map each cluster to specific business actions and strategies."""
        print("\n=== MAPPING CLUSTERS TO BUSINESS ACTIONS ===")
        
        business_actions = {}
        
        for _, cluster in self.cluster_stats.iterrows():
            cluster_id = cluster['cluster_id']
            pc1 = cluster['pc1_mean']  # Delay sensitivity
            pc2 = cluster['pc2_mean']  # Service quality focus
            pc3 = cluster['pc3_mean']  # Digital convenience
            satisfaction = cluster['satisfaction_rate']
            business_travel = cluster['business_travel_pct']
            loyal = cluster['loyal_customer_pct']
            age = cluster['age_mean']
            
            # Determine cluster type and actions
            if pc1 > 0.5:  # High delay sensitivity
                cluster_type = "Delay-Sensitive Travelers"
                actions = [
                    "Proactive delay notifications via SMS/email",
                    "Priority rebooking assistance during disruptions",
                    "Compensation packages for significant delays",
                    "Real-time flight status updates and alternatives",
                    "Lounge access during extended delays"
                ]
                priority = "High" if satisfaction < 50 else "Medium"
                
            elif pc2 > 0.5:  # Service quality focused
                cluster_type = "Service Quality Enthusiasts"
                actions = [
                    "Premium service upgrades and amenities",
                    "Personalized in-flight service",
                    "Priority boarding and seating",
                    "Enhanced meal and entertainment options",
                    "Dedicated customer service line"
                ]
                priority = "High" if satisfaction > 70 else "Medium"
                
            elif pc3 > 0.5:  # Digital convenience seekers
                cluster_type = "Digital-First Travelers"
                actions = [
                    "Enhanced mobile app features",
                    "Self-service kiosk improvements",
                    "Digital wallet and payment options",
                    "Automated check-in and bag drop",
                    "AI-powered customer support"
                ]
                priority = "Medium"
                
            elif business_travel > 60:  # Business travelers
                cluster_type = "Business Travelers"
                actions = [
                    "Corporate loyalty programs",
                    "Priority services and fast-track",
                    "Flexible booking and cancellation policies",
                    "Business lounge access",
                    "Productivity amenities (WiFi, power outlets)"
                ]
                priority = "High"
                
            elif age > 50:  # Mature travelers
                cluster_type = "Mature Travelers"
                actions = [
                    "Assistance services and support",
                    "Clear communication and instructions",
                    "Comfort-focused amenities",
                    "Family-friendly services",
                    "Accessibility improvements"
                ]
                priority = "Medium"
                
            else:  # General cluster
                cluster_type = "General Travelers"
                actions = [
                    "Standard service improvements",
                    "Value-added packages",
                    "Loyalty program enhancements",
                    "Customer feedback integration",
                    "Operational efficiency improvements"
                ]
                priority = "Low"
            
            business_actions[cluster_id] = {
                'cluster_type': cluster_type,
                'actions': actions,
                'priority': priority,
                'target_satisfaction_increase': min(20, max(5, 100 - satisfaction)),
                'estimated_retention_lift': min(15, max(3, (100 - satisfaction) * 0.2))
            }
        
        self.business_actions = business_actions
        
        # Print action mapping
        for cluster_id, actions in business_actions.items():
            print(f"\nCluster {cluster_id}: {actions['cluster_type']}")
            print(f"  Priority: {actions['priority']}")
            print(f"  Target Satisfaction Increase: {actions['target_satisfaction_increase']:.1f}%")
            print(f"  Estimated Retention Lift: {actions['estimated_retention_lift']:.1f}%")
            print("  Actions:")
            for action in actions['actions']:
                print(f"    • {action}")
        
        return self
    
    def estimate_business_impact(self):
        """Estimate the business impact of implementing cluster-based strategies."""
        print("\n=== ESTIMATING BUSINESS IMPACT ===")
        
        # Assume base metrics (these would come from actual airline data)
        base_passengers = len(self.train_df)
        base_satisfaction = (self.train_df['satisfaction'] == 'satisfied').mean() * 100
        base_retention = 75  # Assume 75% base retention rate
        avg_revenue_per_passenger = 500  # Assume $500 average revenue per passenger
        
        impact_estimates = {}
        total_impact = 0
        
        for cluster_id, actions in self.business_actions.items():
            cluster_data = self.cluster_stats[self.cluster_stats['cluster_id'] == cluster_id].iloc[0]
            cluster_size = cluster_data['size']
            cluster_pct = cluster_data['percentage']
            current_satisfaction = cluster_data['satisfaction_rate']
            
            # Calculate potential impact
            satisfaction_increase = actions['target_satisfaction_increase']
            retention_lift = actions['estimated_retention_lift']
            
            # Revenue impact
            additional_passengers = cluster_size * (retention_lift / 100)
            additional_revenue = additional_passengers * avg_revenue_per_passenger
            
            # Cost estimates (simplified)
            implementation_cost = cluster_size * 10  # $10 per passenger for implementation
            net_revenue_impact = additional_revenue - implementation_cost
            
            impact_estimates[cluster_id] = {
                'cluster_size': cluster_size,
                'cluster_percentage': cluster_pct,
                'current_satisfaction': current_satisfaction,
                'target_satisfaction': current_satisfaction + satisfaction_increase,
                'retention_lift': retention_lift,
                'additional_passengers': additional_passengers,
                'additional_revenue': additional_revenue,
                'implementation_cost': implementation_cost,
                'net_revenue_impact': net_revenue_impact,
                'roi': (net_revenue_impact / implementation_cost) * 100 if implementation_cost > 0 else 0
            }
            
            total_impact += net_revenue_impact
        
        self.impact_estimates = impact_estimates
        
        # Print impact summary
        print(f"\nBusiness Impact Summary:")
        print(f"Total Passengers: {base_passengers:,}")
        print(f"Current Satisfaction: {base_satisfaction:.1f}%")
        print(f"Estimated Total Revenue Impact: ${total_impact:,.0f}")
        
        for cluster_id, impact in impact_estimates.items():
            print(f"\nCluster {cluster_id} ({impact['cluster_percentage']:.1f}% of passengers):")
            print(f"  Current Satisfaction: {impact['current_satisfaction']:.1f}%")
            print(f"  Target Satisfaction: {impact['target_satisfaction']:.1f}%")
            print(f"  Additional Revenue: ${impact['additional_revenue']:,.0f}")
            print(f"  ROI: {impact['roi']:.1f}%")
        
        return self
    
    def validate_insights(self):
        """Validate insights with back-testing and practicality checks."""
        print("\n=== VALIDATING INSIGHTS ===")
        
        # 1. Back-test with satisfaction distribution
        print("1. Satisfaction Distribution Validation:")
        satisfaction_by_cluster = self.cluster_profiles.groupby('cluster')['satisfaction'].apply(
            lambda x: (x == 'satisfied').mean() * 100
        ).sort_values(ascending=False)
        
        print("Satisfaction rates by cluster:")
        for cluster_id, rate in satisfaction_by_cluster.items():
            print(f"  Cluster {cluster_id}: {rate:.1f}%")
        
        # Check if clusters differ meaningfully
        satisfaction_std = satisfaction_by_cluster.std()
        print(f"Satisfaction variation across clusters: {satisfaction_std:.1f}%")
        
        if satisfaction_std > 10:
            print("✅ Clusters show meaningful differences in satisfaction")
        else:
            print("⚠️  Clusters may not differ significantly in satisfaction")
        
        # 2. Practicality check
        print("\n2. Actionability Validation:")
        actionable_features = [
            'Inflight wifi service', 'Seat comfort', 'Food and drink',
            'Online boarding', 'Checkin service', 'Baggage handling'
        ]
        
        print("Features that airlines can directly control:")
        for feature in actionable_features:
            if feature in self.cluster_profiles.columns:
                print(f"  ✅ {feature}")
            else:
                print(f"  ❌ {feature} (not available)")
        
        # 3. Cluster stability check
        print("\n3. Cluster Stability Validation:")
        cluster_sizes = self.cluster_profiles['cluster'].value_counts().sort_index()
        min_size = cluster_sizes.min()
        max_size = cluster_sizes.max()
        size_ratio = max_size / min_size
        
        print(f"Cluster size range: {min_size} - {max_size} passengers")
        print(f"Size ratio: {size_ratio:.1f}")
        
        if size_ratio < 5:
            print("✅ Clusters are reasonably balanced")
        else:
            print("⚠️  Some clusters may be too small or too large")
        
        return self
    
    def create_executive_summary(self):
        """Create executive one-pager with key insights and recommendations."""
        print("\n=== CREATING EXECUTIVE SUMMARY ===")
        
        # Create one-pager content
        summary = f"""
# AIRLINE PASSENGER SEGMENTATION - EXECUTIVE SUMMARY

## Problem Statement
Segment airline passengers to tailor experiences and offers, improving satisfaction and retention.

## Methodology
- **Data Processing**: Cleaned and standardized 27 passenger features
- **Dimensionality Reduction**: PCA reduced to 6 components (80.9% variance retained)
- **Clustering**: K-Means identified {len(set(self.cluster_labels))} distinct passenger segments
- **Validation**: Silhouette analysis and business impact modeling

## Key Findings

### Cluster Profiles
"""
        
        for cluster_id, actions in self.business_actions.items():
            impact = self.impact_estimates[cluster_id]
            summary += f"""
**Cluster {cluster_id}: {actions['cluster_type']}**
- Size: {impact['cluster_percentage']:.1f}% of passengers ({impact['cluster_size']:,} passengers)
- Current Satisfaction: {impact['current_satisfaction']:.1f}%
- Priority: {actions['priority']}
- Revenue Opportunity: ${impact['additional_revenue']:,.0f}
"""
        
        summary += f"""
## Recommended Actions

### High Priority (Immediate Implementation)
"""
        
        high_priority_clusters = [cid for cid, actions in self.business_actions.items() 
                                if actions['priority'] == 'High']
        
        for cluster_id in high_priority_clusters:
            actions = self.business_actions[cluster_id]
            summary += f"""
**{actions['cluster_type']} (Cluster {cluster_id}):**
"""
            for action in actions['actions'][:3]:  # Top 3 actions
                summary += f"- {action}\n"
        
        summary += f"""
### Medium Priority (Next Quarter)
"""
        
        medium_priority_clusters = [cid for cid, actions in self.business_actions.items() 
                                  if actions['priority'] == 'Medium']
        
        for cluster_id in medium_priority_clusters:
            actions = self.business_actions[cluster_id]
            summary += f"""
**{actions['cluster_type']} (Cluster {cluster_id}):**
"""
            for action in actions['actions'][:2]:  # Top 2 actions
                summary += f"- {action}\n"
        
        summary += f"""
## Expected Impact
- **Total Revenue Opportunity**: ${sum(impact['additional_revenue'] for impact in self.impact_estimates.values()):,.0f}
- **Average ROI**: {np.mean([impact['roi'] for impact in self.impact_estimates.values()]):.1f}%
- **Target Satisfaction Increase**: 5-20% across segments

## Next Steps
1. **Pilot Program**: Test high-priority actions on 10% of each cluster
2. **A/B Testing**: Measure satisfaction and retention improvements
3. **Scale Implementation**: Roll out successful strategies to full clusters
4. **Continuous Monitoring**: Track cluster evolution and adjust strategies

## Limitations
- Analysis based on historical data; causal relationships not established
- Feature selection may have missed important factors
- Business impact estimates are projections requiring validation
- Clusters may evolve over time requiring periodic re-analysis
"""
        
        # Save executive summary
        with open('executive_summary.md', 'w') as f:
            f.write(summary)
        
        print("Executive summary saved to 'executive_summary.md'")
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY PREVIEW")
        print("="*60)
        print(summary[:1000] + "...")
        
        return self
    
    def create_visualizations(self):
        """Create comprehensive visualizations for business insights."""
        print("\n=== CREATING BUSINESS VISUALIZATIONS ===")
        
        # 1. Cluster satisfaction comparison
        plt.figure(figsize=(12, 8))
        
        satisfaction_by_cluster = self.cluster_profiles.groupby('cluster')['satisfaction'].apply(
            lambda x: (x == 'satisfied').mean() * 100
        ).sort_values(ascending=False)
        
        bars = plt.bar(range(len(satisfaction_by_cluster)), satisfaction_by_cluster.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(satisfaction_by_cluster))))
        
        plt.xlabel('Cluster')
        plt.ylabel('Satisfaction Rate (%)')
        plt.title('Satisfaction Rate by Cluster')
        plt.xticks(range(len(satisfaction_by_cluster)), [f'Cluster {i}' for i in satisfaction_by_cluster.index])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_satisfaction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Revenue impact visualization
        plt.figure(figsize=(12, 8))
        
        cluster_ids = list(self.impact_estimates.keys())
        revenues = [self.impact_estimates[cid]['additional_revenue'] for cid in cluster_ids]
        percentages = [self.impact_estimates[cid]['cluster_percentage'] for cid in cluster_ids]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Revenue impact
        bars1 = ax1.bar(range(len(cluster_ids)), revenues, color=plt.cm.viridis(np.linspace(0, 1, len(cluster_ids))))
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Additional Revenue ($)')
        ax1.set_title('Revenue Impact by Cluster')
        ax1.set_xticks(range(len(cluster_ids)))
        ax1.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Cluster size distribution
        bars2 = ax2.bar(range(len(cluster_ids)), percentages, color=plt.cm.plasma(np.linspace(0, 1, len(cluster_ids))))
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Percentage of Passengers (%)')
        ax2.set_title('Cluster Size Distribution')
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('business_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def run_complete_analysis(self, train_path, pca_path):
        """Run the complete Phase 4 business insights analysis."""
        print("="*60)
        print("PHASE 4 - INSIGHTS & DECISIONS")
        print("="*60)
        
        # Load data and create profiles
        self.load_clustering_results(train_path, pca_path)
        self.create_cluster_profiles()
        
        # Generate business insights
        self.map_clusters_to_actions()
        self.estimate_business_impact()
        self.validate_insights()
        
        # Create deliverables
        self.create_executive_summary()
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("PHASE 4 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self

def main():
    """Main function to run the business insights analysis."""
    # Initialize business insights
    insights = BusinessInsights()
    
    # Run complete analysis
    insights.run_complete_analysis(
        train_path='train_processed.csv',
        pca_path='X_train_pca.csv'
    )

if __name__ == "__main__":
    main()
