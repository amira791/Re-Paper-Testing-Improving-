"""
Data Quality Monitor - Track and visualize data quality improvements
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_data_quality(before_path, after_path):
    """Compare data quality before and after cleaning"""
    
    # Load datasets
    df_before = pd.read_parquet(before_path)
    df_after = pd.read_parquet(after_path)
    
    print("="*80)
    print("DATA QUALITY COMPARISON: BEFORE vs AFTER CLEANING")
    print("="*80)
    
    # Create comparison report
    comparison = []
    
    # 1. Missing values comparison
    missing_before = df_before.isnull().sum().sum()
    missing_after = df_after.isnull().sum().sum()
    missing_reduction = ((missing_before - missing_after) / missing_before * 100) if missing_before > 0 else 0
    
    comparison.append({
        'Metric': 'Total Missing Values',
        'Before': f"{missing_before:,}",
        'After': f"{missing_after:,}",
        'Improvement': f"{missing_reduction:.1f}% reduction",
        'Status': '✅' if missing_reduction > 50 else '⚠️'
    })
    
    # 2. Temperature outliers
    temp_cols = ['bcell_maxTemp', 'bcell_minTemp']
    for col in temp_cols:
        if col in df_before.columns and col in df_after.columns:
            # Count unrealistic values
            outliers_before = ((df_before[col] < -20) | (df_before[col] > 60)).sum()
            outliers_after = ((df_after[col] < -20) | (df_after[col] > 60)).sum()
            
            if outliers_before > 0:
                reduction = ((outliers_before - outliers_after) / outliers_before * 100)
                comparison.append({
                    'Metric': f'{col} Outliers',
                    'Before': f"{outliers_before:,}",
                    'After': f"{outliers_after:,}",
                    'Improvement': f"{reduction:.1f}% reduction",
                    'Status': '✅' if outliers_after == 0 else '⚠️'
                })
    
    # 3. SOH boundary clustering
    if 'soh_capacity' in df_before.columns and 'soh_capacity' in df_after.columns:
        boundary_before = ((df_before['soh_capacity'] == 70) | (df_before['soh_capacity'] == 105)).sum()
        boundary_after = ((df_after['soh_capacity'] == 70) | (df_after['soh_capacity'] == 105)).sum()
        
        reduction = ((boundary_before - boundary_after) / boundary_before * 100) if boundary_before > 0 else 0
        comparison.append({
            'Metric': 'SOH Boundary Values (70 or 105)',
            'Before': f"{boundary_before:,}",
            'After': f"{boundary_after:,}",
            'Improvement': f"{reduction:.1f}% reduction",
            'Status': '✅' if reduction > 50 else '⚠️'
        })
    
    # 4. Time series issues
    if 'time' in df_before.columns and 'time_diff' in df_after.columns:
        # Check for negative time differences (before cleaning implied by huge std)
        time_std_before = df_before['time'].diff().std()
        negative_after = (df_after['time_diff'] < 0).sum() if 'time_diff' in df_after.columns else 'N/A'
        
        comparison.append({
            'Metric': 'Time Series Issues',
            'Before': f"Std: {time_std_before:,.0f}",
            'After': f"Negative diffs: {negative_after:,}",
            'Improvement': 'Session-based time created',
            'Status': '✅' if negative_after == 0 else '⚠️'
        })
    
    # 5. Vehicle balance
    if 'vehicle_id' in df_before.columns and 'vehicle_id' in df_after.columns:
        counts_before = df_before['vehicle_id'].value_counts()
        counts_after = df_after['vehicle_id'].value_counts()
        
        imbalance_before = counts_before.max() / counts_before.min()
        imbalance_after = counts_after.max() / counts_after.min()
        
        comparison.append({
            'Metric': 'Vehicle Imbalance Ratio',
            'Before': f"{imbalance_before:.1f}x",
            'After': f"{imbalance_after:.1f}x",
            'Improvement': f"Reduced by {((imbalance_before - imbalance_after)/imbalance_before*100):.1f}%",
            'Status': '✅' if imbalance_after < 10 else '⚠️'
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))
    
    # Visualize improvements
    create_quality_visualizations(comparison_df, df_before, df_after)
    
    return comparison_df

def create_quality_visualizations(comparison_df, df_before, df_after):
    """Create visualizations of data quality improvements"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Quality Improvements After Cleaning', fontsize=16, fontweight='bold')
    
    # 1. Missing values comparison
    ax1 = axes[0, 0]
    missing_cols = ['bcell_maxTemp', 'bcell_minTemp', 'bcell_maxVoltage', 'bcell_minVoltage']
    
    missing_before = []
    missing_after = []
    labels = []
    
    for col in missing_cols:
        if col in df_before.columns and col in df_after.columns:
            missing_before.append(df_before[col].isna().sum() / len(df_before) * 100)
            missing_after.append(df_after[col].isna().sum() / len(df_after) * 100)
            labels.append(col)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, missing_before, width, label='Before', alpha=0.7, color='red')
    ax1.bar(x + width/2, missing_after, width, label='After', alpha=0.7, color='green')
    
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Missing Values (%)')
    ax1.set_title('Missing Values Reduction')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature distribution before/after
    ax2 = axes[0, 1]
    if 'bcell_maxTemp' in df_before.columns and 'bcell_maxTemp' in df_after.columns:
        ax2.hist(df_before['bcell_maxTemp'].dropna(), bins=50, alpha=0.5, label='Before', color='red', density=True)
        ax2.hist(df_after['bcell_maxTemp'].dropna(), bins=50, alpha=0.5, label='After', color='green', density=True)
        
        ax2.set_xlabel('Max Temperature (°C)')
        ax2.set_ylabel('Density')
        ax2.set_title('Temperature Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add realistic range lines
        ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axvline(x=50, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # 3. SOH distribution before/after
    ax3 = axes[0, 2]
    if 'soh_capacity' in df_before.columns and 'soh_capacity' in df_after.columns:
        ax3.hist(df_before['soh_capacity'], bins=50, alpha=0.5, label='Before', color='red', density=True)
        ax3.hist(df_after['soh_capacity'], bins=50, alpha=0.5, label='After', color='green', density=True)
        
        ax3.set_xlabel('SOH (%)')
        ax3.set_ylabel('Density')
        ax3.set_title('SOH Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Vehicle distribution
    ax4 = axes[1, 0]
    if 'vehicle_id' in df_before.columns and 'vehicle_id' in df_after.columns:
        vehicle_counts_before = df_before['vehicle_id'].value_counts().sort_index()
        vehicle_counts_after = df_after['vehicle_id'].value_counts().sort_index()
        
        x = np.arange(len(vehicle_counts_before))
        width = 0.35
        
        ax4.bar(x - width/2, vehicle_counts_before.values, width, label='Before', alpha=0.7, color='red')
        ax4.bar(x + width/2, vehicle_counts_after.values, width, label='After', alpha=0.7, color='green')
        
        ax4.set_xlabel('Vehicle ID')
        ax4.set_ylabel('Number of Rows')
        ax4.set_title('Vehicle Distribution')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'V{vid}' for vid in vehicle_counts_before.index])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. New features created
    ax5 = axes[1, 1]
    new_features = ['cell_imbalance', 'temp_gradient', 'power_kw', 'soc_category', 'driving_intensity']
    
    available_features = [feat for feat in new_features if feat in df_after.columns]
    
    if available_features:
        y_pos = np.arange(len(available_features))
        presence = [1] * len(available_features)  # All present in after
        
        ax5.barh(y_pos, presence, color='green', alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(available_features)
        ax5.set_xlabel('Presence (1 = Present)')
        ax5.set_title('New Features Created')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Data quality score
    ax6 = axes[1, 2]
    
    # Calculate quality scores (simplified)
    quality_metrics = {
        'Missing Data': 100 - (df_after.isnull().sum().sum() / (df_after.shape[0] * df_after.shape[1]) * 100),
        'Temp Range': 100 if ((df_after['bcell_maxTemp'] >= -20) & (df_after['bcell_maxTemp'] <= 60)).all() else 50,
        'SOH Spread': 100 - ((df_after['soh_capacity'] == 70).sum() + (df_after['soh_capacity'] == 105).sum()) / len(df_after) * 100,
        'Time Integrity': 100 if 'time_diff' not in df_after.columns or (df_after['time_diff'] < 0).sum() == 0 else 50,
        'Vehicle Balance': 100 / (df_after['vehicle_id'].value_counts().max() / df_after['vehicle_id'].value_counts().min())
    }
    
    metrics = list(quality_metrics.keys())
    scores = list(quality_metrics.values())
    
    colors = ['green' if score > 80 else 'orange' if score > 60 else 'red' for score in scores]
    
    ax6.barh(metrics, scores, color=colors, alpha=0.7)
    ax6.set_xlabel('Quality Score (%)')
    ax6.set_title('Data Quality Metrics')
    ax6.set_xlim(0, 110)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (score, color) in enumerate(zip(scores, colors)):
        ax6.text(score + 1, i, f'{score:.0f}%', va='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_quality_improvements.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary
    summary = {
        'Total Rows Before': len(df_before),
        'Total Rows After': len(df_after),
        'Total Columns Before': len(df_before.columns),
        'Total Columns After': len(df_after.columns),
        'Missing Reduction (%)': (df_before.isnull().sum().sum() - df_after.isnull().sum().sum()) / df_before.isnull().sum().sum() * 100 if df_before.isnull().sum().sum() > 0 else 0,
        'Overall Quality Score': np.mean(list(quality_metrics.values()))
    }
    
    with open('data_quality_summary.txt', 'w') as f:
        f.write("DATA QUALITY SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")

def main():
    """Main quality monitoring function"""
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    before_file = data_path / "all_vehicles_combined.parquet"
    after_file = data_path / "all_vehicles_cleaned.parquet"
    
    if before_file.exists() and after_file.exists():
        comparison = compare_data_quality(before_file, after_file)
        print("\n Data quality comparison complete!")
        print(" Visualization saved as 'data_quality_improvements.png'")
        print(" Summary saved as 'data_quality_summary.txt'")
        return comparison
    else:
        print(" Required files not found!")
        print(f"Expected: {before_file}")
        print(f"Expected: {after_file}")
        return None

if __name__ == "__main__":
    comparison_results = main()