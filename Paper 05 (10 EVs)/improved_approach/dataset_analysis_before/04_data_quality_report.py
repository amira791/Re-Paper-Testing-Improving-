# file: 04_data_quality_report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(B)")
OUTPUT_PATH = Path("./analysis_output/data_quality")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

def check_sensor_anomalies(df, vehicle_num):
    """Check for sensor anomalies and data quality issues"""
    anomalies = {}
    
    # Standardize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Check for sentinel values (like 65535)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            # Check for extreme values that might be sentinel/error codes
            if df[col].max() >= 65535:
                anomalies[f'{col}_65535_sentinel'] = (df[col] >= 65535).sum()
            
            # Check for zero values where not expected
            if 'voltage' in col and (df[col] == 0).sum() > 0:
                anomalies[f'{col}_zero_values'] = (df[col] == 0).sum()
            
            # Check for negative values where not expected
            if 'voltage' in col and 'temp' not in col and (df[col] < 0).sum() > 0:
                anomalies[f'{col}_negative_values'] = (df[col] < 0).sum()
    
    # Check for SOC anomalies
    soc_cols = [col for col in df.columns if 'soc' in col]
    if soc_cols:
        soc_col = soc_cols[0]
        # SOC should be between 0 and 100
        invalid_soc = ((df[soc_col] < 0) | (df[soc_col] > 100)).sum()
        if invalid_soc > 0:
            anomalies['soc_out_of_range'] = invalid_soc
    
    # Check for temperature anomalies
    temp_cols = [col for col in df.columns if 'temp' in col]
    for temp_col in temp_cols:
        # Temperature should be reasonable for EV batteries (-20 to 60°C)
        extreme_temp = ((df[temp_col] < -20) | (df[temp_col] > 60)).sum()
        if extreme_temp > 0:
            anomalies[f'{temp_col}_extreme'] = extreme_temp
    
    # Check for voltage consistency
    if 'bcell_maxvoltage' in df.columns and 'bcell_minvoltage' in df.columns:
        # Max voltage should be >= min voltage
        invalid_voltage_pairs = (df['bcell_maxvoltage'] < df['bcell_minvoltage']).sum()
        if invalid_voltage_pairs > 0:
            anomalies['max_min_voltage_inconsistency'] = invalid_voltage_pairs
        
        # Calculate voltage spread
        voltage_spread = df['bcell_maxvoltage'] - df['bcell_minvoltage']
        # Large spreads might indicate cell imbalance
        large_spread = (voltage_spread > 0.5).sum()  # More than 0.5V spread
        if large_spread > 0:
            anomalies['large_voltage_spread'] = large_spread
    
    # Check timestamp consistency
    time_cols = [col for col in df.columns if 'time' in col]
    if time_cols:
        time_col = time_cols[0]
        time_diffs = df[time_col].diff().dropna()
        
        if len(time_diffs) > 0:
            expected_interval = 10  # 10 seconds for 0.1 Hz sampling
            time_anomalies = (np.abs(time_diffs - expected_interval) > 1).sum()
            if time_anomalies > 0:
                anomalies['timestamp_inconsistencies'] = time_anomalies
    
    return anomalies

def check_missing_patterns(df, vehicle_num):
    """Check patterns in missing data"""
    missing_info = {}
    
    # Overall missing values
    total_missing = df.isnull().sum().sum()
    missing_info['total_missing'] = total_missing
    missing_info['missing_percentage'] = (total_missing / (len(df) * len(df.columns))) * 100
    
    # Missing by column
    missing_by_column = df.isnull().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]
    if len(missing_by_column) > 0:
        missing_info['columns_with_missing'] = missing_by_column.to_dict()
    
    # Check for consecutive missing values (gaps)
    for col in df.columns:
        if df[col].isnull().any():
            # Find gaps
            is_null = df[col].isnull()
            null_groups = (is_null != is_null.shift()).cumsum()
            
            for group_id, group in df.groupby(null_groups):
                if is_null.iloc[group.index[0]]:
                    gap_length = len(group)
                    if gap_length > 10:  # Significant gap
                        key = f'{col}_gap_{gap_length}'
                        missing_info[key] = missing_info.get(key, 0) + 1
    
    return missing_info

def analyze_data_distributions(df, vehicle_num):
    """Analyze data distributions for key variables"""
    distributions = {}
    
    key_columns = [
        'vhc_speed', 'hv_current', 'bcell_soc', 
        'bcell_maxvoltage', 'bcell_minvoltage',
        'bcell_maxtemp', 'bcell_mintemp'
    ]
    
    available_cols = [col for col in key_columns if col in df.columns]
    
    for col in available_cols:
        if df[col].dtype in [np.float64, np.int64]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                distributions[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'zeros_pct': (col_data == 0).sum() / len(col_data) * 100,
                    'outliers_3sigma': ((col_data - col_data.mean()).abs() > 3 * col_data.std()).sum()
                }
    
    return distributions

def generate_quality_report(vehicle_num):
    """Generate comprehensive data quality report for a vehicle"""
    print(f"  Generating quality report for Vehicle #{vehicle_num}...")
    
    try:
        # Load data
        file_path = DATA_PATH / f"vehicle#{vehicle_num}.xlsx"
        df = pd.read_excel(file_path)
        
        report = {
            'vehicle': vehicle_num,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns)
        }
        
        # Check anomalies
        anomalies = check_sensor_anomalies(df, vehicle_num)
        report['anomalies'] = anomalies
        report['total_anomalies'] = sum(anomalies.values())
        
        # Check missing patterns
        missing_info = check_missing_patterns(df, vehicle_num)
        report['missing_info'] = missing_info
        
        # Analyze distributions
        distributions = analyze_data_distributions(df, vehicle_num)
        report['distributions'] = distributions
        
        # Calculate quality score
        quality_score = 100
        
        # Deduct for anomalies
        if report['total_anomalies'] > 0:
            anomaly_penalty = min(50, (report['total_anomalies'] / len(df)) * 10000)
            quality_score -= anomaly_penalty
        
        # Deduct for missing data
        if 'missing_percentage' in missing_info:
            quality_score -= missing_info['missing_percentage'] * 2
        
        report['quality_score'] = max(0, quality_score)
        
        return report
        
    except Exception as e:
        print(f"    Error analyzing Vehicle #{vehicle_num}: {e}")
        return None

def plot_quality_report(all_reports):
    """Create visualizations for data quality report"""
    
    # Prepare data for plotting
    vehicles = []
    quality_scores = []
    anomaly_counts = []
    missing_percentages = []
    
    for vehicle_num, report in all_reports.items():
        if report:
            vehicles.append(vehicle_num)
            quality_scores.append(report['quality_score'])
            anomaly_counts.append(report.get('total_anomalies', 0))
            missing_percentages.append(report.get('missing_info', {}).get('missing_percentage', 0))
    
    if not vehicles:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Quality scores
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(vehicles)), quality_scores, 
                   color=['green' if score >= 80 else 'orange' if score >= 60 else 'red' 
                          for score in quality_scores])
    
    # Add score labels
    for bar, score in zip(bars, quality_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom')
    
    ax1.set_title('Data Quality Scores by Vehicle')
    ax1.set_xlabel('Vehicle')
    ax1.set_ylabel('Quality Score (0-100)')
    ax1.set_xticks(range(len(vehicles)))
    ax1.set_xticklabels([f'V{v}' for v in vehicles])
    ax1.set_ylim(0, 105)
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (≥80)')
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Fair (≥60)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Anomaly counts
    ax2 = axes[0, 1]
    ax2.bar(range(len(vehicles)), anomaly_counts, color='salmon')
    ax2.set_title('Total Anomalies Detected')
    ax2.set_xlabel('Vehicle')
    ax2.set_ylabel('Number of Anomalies')
    ax2.set_xticks(range(len(vehicles)))
    ax2.set_xticklabels([f'V{v}' for v in vehicles])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Missing data percentages
    ax3 = axes[1, 0]
    ax3.bar(range(len(vehicles)), missing_percentages, color='skyblue')
    ax3.set_title('Missing Data Percentage')
    ax3.set_xlabel('Vehicle')
    ax3.set_ylabel('Missing Data (%)')
    ax3.set_xticks(range(len(vehicles)))
    ax3.set_xticklabels([f'V{v}' for v in vehicles])
    ax3.set_ylim(0, max(missing_percentages) * 1.2 if max(missing_percentages) > 0 else 5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Common anomaly types
    ax4 = axes[1, 1]
    
    # Collect all anomaly types
    anomaly_types = {}
    for vehicle_num, report in all_reports.items():
        if report and 'anomalies' in report:
            for anomaly_type, count in report['anomalies'].items():
                if count > 0:
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + count
    
    if anomaly_types:
        # Get top 10 anomaly types
        top_anomalies = sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True)[:10]
        anomaly_names = [name for name, _ in top_anomalies]
        anomaly_counts = [count for _, count in top_anomalies]
        
        bars = ax4.barh(range(len(anomaly_names)), anomaly_counts, color='lightcoral')
        ax4.set_yticks(range(len(anomaly_names)))
        ax4.set_yticklabels(anomaly_names, fontsize=9)
        ax4.set_title('Top 10 Most Common Anomaly Types')
        ax4.set_xlabel('Total Occurrences Across All Vehicles')
        ax4.invert_yaxis()  # Highest at top
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for bar, count in zip(bars, anomaly_counts):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {count}', va='center')
    else:
        ax4.text(0.5, 0.5, 'No anomalies detected', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Anomaly Types')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'data_quality_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual vehicle anomaly plots
    for vehicle_num, report in all_reports.items():
        if report and report.get('anomalies'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            anomalies = report['anomalies']
            # Filter for anomalies with counts > 0
            valid_anomalies = {k: v for k, v in anomalies.items() if v > 0}
            
            if valid_anomalies:
                names = list(valid_anomalies.keys())
                counts = list(valid_anomalies.values())
                
                bars = ax.bar(range(len(names)), counts, color='lightcoral')
                ax.set_title(f'Vehicle #{vehicle_num} - Detected Anomalies')
                ax.set_xlabel('Anomaly Type')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(OUTPUT_PATH / f'vehicle{vehicle_num}_anomalies.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()

def main():
    """Main function for data quality analysis"""
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    all_reports = {}
    
    # Analyze each vehicle
    for vehicle_num in range(1, 11):
        report = generate_quality_report(vehicle_num)
        all_reports[vehicle_num] = report
        
        if report:
            print(f"\n  Vehicle #{vehicle_num} Quality Report:")
            print(f"    Rows: {report['total_rows']:,}")
            print(f"    Columns: {report['total_columns']}")
            print(f"    Quality Score: {report['quality_score']:.1f}/100")
            print(f"    Total Anomalies: {report['total_anomalies']:,}")
            print(f"    Missing Data: {report['missing_info'].get('missing_percentage', 0):.2f}%")
            
            # List top anomalies
            if report['anomalies']:
                print(f"    Top Anomalies:")
                sorted_anomalies = sorted(report['anomalies'].items(), key=lambda x: x[1], reverse=True)[:3]
                for anomaly_type, count in sorted_anomalies:
                    if count > 0:
                        print(f"      - {anomaly_type}: {count:,}")
    
    # Save detailed reports
    for vehicle_num, report in all_reports.items():
        if report:
            # Convert to DataFrame-friendly format
            report_df = pd.DataFrame({
                'metric': list(report.keys()),
                'value': list(report.values())
            })
            report_df.to_csv(OUTPUT_PATH / f'vehicle{vehicle_num}_quality_report.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_quality_report(all_reports)
    print(f"Visualizations saved to: {OUTPUT_PATH}")
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*60}")
    
    summary_data = []
    for vehicle_num, report in all_reports.items():
        if report:
            summary_data.append({
                'vehicle': vehicle_num,
                'rows': report['total_rows'],
                'quality_score': report['quality_score'],
                'anomalies': report['total_anomalies'],
                'missing_pct': report['missing_info'].get('missing_percentage', 0),
                'quality_rating': 'Good' if report['quality_score'] >= 80 else 
                                 'Fair' if report['quality_score'] >= 60 else 
                                 'Poor'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_PATH / 'data_quality_summary.csv', index=False)
        
        print(f"\nOverall Data Quality Assessment:")
        print(f"  Average Quality Score: {summary_df['quality_score'].mean():.1f}/100")
        print(f"  Total Anomalies: {summary_df['anomalies'].sum():,}")
        print(f"  Average Missing Data: {summary_df['missing_pct'].mean():.2f}%")
        
        print(f"\nVehicle Rankings by Quality:")
        summary_df = summary_df.sort_values('quality_score', ascending=False)
        for idx, row in summary_df.iterrows():
            print(f"  {row['quality_rating']}: Vehicle #{row['vehicle']} ({row['quality_score']:.1f})")
        
        print(f"\nRecommendations:")
        
        # Identify vehicles with poor quality
        poor_quality = summary_df[summary_df['quality_score'] < 60]
        if len(poor_quality) > 0:
            print(f"  Vehicles requiring data cleaning: {list(poor_quality['vehicle'])}")
        
        # Identify vehicles with many anomalies
        high_anomaly = summary_df[summary_df['anomalies'] > 1000]
        if len(high_anomaly) > 0:
            print(f"  Vehicles with high anomaly counts: {list(high_anomaly['vehicle'])}")

if __name__ == "__main__":
    main()