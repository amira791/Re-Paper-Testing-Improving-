# file: 01_data_summary.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths and style
DATA_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(B)")
OUTPUT_PATH = Path(r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\improved_approach\dataset_analysis_before\results_before')
OUTPUT_PATH.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_vehicle_data(vehicle_num):
    """Load a vehicle's Excel file"""
    file_path = DATA_PATH / f"vehicle#{vehicle_num}.xlsx"
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded Vehicle#{vehicle_num}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        # Try CSV or other formats
        for ext in ['.csv', '.xls']:
            alt_path = DATA_PATH / f"vehicle#{vehicle_num}{ext}"
            if alt_path.exists():
                df = pd.read_csv(alt_path) if ext == '.csv' else pd.read_excel(alt_path)
                print(f"Loaded Vehicle#{vehicle_num}: {len(df)} rows, {len(df.columns)} columns")
                return df
        print(f"File for Vehicle#{vehicle_num} not found")
        return None

def get_basic_stats(df, vehicle_num):
    """Get basic statistics for a vehicle"""
    stats = {
        'vehicle': vehicle_num,
        'rows': len(df),
        'columns': len(df.columns),
        'date_range': None,
        'missing_values': df.isnull().sum().sum(),
        'columns_list': list(df.columns)
    }
    
    # Check for time column
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    if time_cols:
        time_col = time_cols[0]
        if df[time_col].dtype in ['int64', 'float64']:
            stats['time_min'] = df[time_col].min()
            stats['time_max'] = df[time_col].max()
            stats['time_range'] = stats['time_max'] - stats['time_min']
            # Estimate sampling frequency
            if len(df) > 1:
                time_diff = df[time_col].diff().median()
                stats['sampling_interval'] = time_diff
                stats['estimated_freq_hz'] = 1 / time_diff if time_diff > 0 else 0
    
    return stats

def analyze_numeric_columns(df, vehicle_num):
    """Analyze numeric columns for each vehicle"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    analysis = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            analysis[col] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'missing': df[col].isnull().sum(),
                'zeros': (col_data == 0).sum(),
                'negative': (col_data < 0).sum() if col_data.dtype in ['float64', 'int64'] else 0
            }
    
    # Save to file
    stats_df = pd.DataFrame(analysis).T
    stats_df.to_csv(OUTPUT_PATH / f'vehicle{vehicle_num}_numeric_stats.csv')
    
    return analysis

def detect_charging_segments(df, vehicle_num):
    """Detect potential charging segments"""
    charging_info = {}
    
    # Check available columns
    cols_lower = [col.lower() for col in df.columns]
    
    # Look for charging signal
    charging_cols = [col for col in df.columns if 'charging' in col.lower()]
    
    if charging_cols:
        charging_col = charging_cols[0]
        charging_info['charging_col'] = charging_col
        charging_info['unique_values'] = df[charging_col].unique().tolist()
        charging_info['value_counts'] = df[charging_col].value_counts().to_dict()
        
        # Detect charging mode based on values
        # Common patterns: 1=charging, 3=driving, or 0=charging
        unique_vals = df[charging_col].unique()
        charging_info['potential_charging_vals'] = []
        
        # Heuristic: values that appear when current is negative (if current col exists)
        current_cols = [col for col in df.columns if 'current' in col.lower()]
        if current_cols:
            current_col = current_cols[0]
            for val in unique_vals:
                mask = df[charging_col] == val
                avg_current = df.loc[mask, current_col].mean()
                # If negative current (charging) is predominant for this charging signal value
                if avg_current < -1:  # Threshold for charging current
                    charging_info['potential_charging_vals'].append(val)
    
    return charging_info

def create_summary_report(all_stats):
    """Create a comprehensive summary report"""
    summary_df = pd.DataFrame(all_stats)
    
    # Calculate some additional metrics
    summary_df['data_duration_days'] = summary_df.apply(
        lambda x: x['time_range'] / (3600 * 24) if pd.notnull(x.get('time_range')) else None, 
        axis=1
    )
    
    # Save summary
    summary_df.to_csv(OUTPUT_PATH / 'vehicle_summary.csv', index=False)
    
    return summary_df

def plot_vehicle_comparison(summary_df):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data points per vehicle
    ax1 = axes[0, 0]
    summary_df.plot(x='vehicle', y='rows', kind='bar', ax=ax1, legend=False)
    ax1.set_title('Number of Data Points per Vehicle')
    ax1.set_ylabel('Rows')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Missing values
    ax2 = axes[0, 1]
    summary_df.plot(x='vehicle', y='missing_values', kind='bar', ax=ax2, legend=False, color='red')
    ax2.set_title('Missing Values per Vehicle')
    ax2.set_ylabel('Missing Values Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Data duration (if available)
    if 'data_duration_days' in summary_df.columns and summary_df['data_duration_days'].notnull().any():
        ax3 = axes[1, 0]
        summary_df.plot(x='vehicle', y='data_duration_days', kind='bar', ax=ax3, legend=False, color='green')
        ax3.set_title('Estimated Data Duration (Days)')
        ax3.set_ylabel('Days')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Column count comparison
    ax4 = axes[1, 1]
    summary_df['num_columns'] = summary_df['columns_list'].apply(len)
    summary_df.plot(x='vehicle', y='num_columns', kind='bar', ax=ax4, legend=False, color='purple')
    ax4.set_title('Number of Columns per Vehicle')
    ax4.set_ylabel('Column Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'vehicle_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    print("=" * 60)
    print("DATASET ANALYSIS - 10 EVs")
    print("=" * 60)
    
    all_stats = []
    all_charging_info = {}
    
    # Analyze each vehicle
    for vehicle_num in range(1, 11):
        print(f"\n{'='*40}")
        print(f"Analyzing Vehicle #{vehicle_num}")
        print(f"{'='*40}")
        
        df = load_vehicle_data(vehicle_num)
        if df is None:
            continue
            
        # Get basic stats
        stats = get_basic_stats(df, vehicle_num)
        all_stats.append(stats)
        
        # Analyze numeric columns
        print(f"Analyzing numeric columns...")
        numeric_analysis = analyze_numeric_columns(df, vehicle_num)
        
        # Detect charging patterns
        print(f"Detecting charging segments...")
        charging_info = detect_charging_segments(df, vehicle_num)
        all_charging_info[f"Vehicle#{vehicle_num}"] = charging_info
        
        # Display key findings
        print(f"\nKey findings for Vehicle #{vehicle_num}:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        if 'time_range' in stats and stats['time_range']:
            days = stats['time_range'] / (3600 * 24)
            print(f"  - Time range: {days:.1f} days")
        
        if charging_info.get('charging_col'):
            print(f"  - Charging column: {charging_info['charging_col']}")
            print(f"  - Unique values: {charging_info.get('unique_values', [])}")
            print(f"  - Potential charging values: {charging_info.get('potential_charging_vals', [])}")
    
    # Create summary report
    print(f"\n{'='*60}")
    print("CREATING SUMMARY REPORT")
    print(f"{'='*60}")
    
    summary_df = create_summary_report(all_stats)
    print(f"Summary saved to: {OUTPUT_PATH / 'vehicle_summary.csv'}")
    
    # Plot comparisons
    plot_vehicle_comparison(summary_df)
    print(f"Comparison plots saved to: {OUTPUT_PATH / 'vehicle_comparison.png'}")
    
    # Save charging info
    charging_df = pd.DataFrame(all_charging_info).T
    charging_df.to_csv(OUTPUT_PATH / 'charging_patterns.csv')
    print(f"Charging patterns saved to: {OUTPUT_PATH / 'charging_patterns.csv'}")
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total vehicles analyzed: {len(summary_df)}")
    print(f"Total data points: {summary_df['rows'].sum():,}")
    print(f"Average rows per vehicle: {summary_df['rows'].mean():,.0f}")
    print(f"Total missing values: {summary_df['missing_values'].sum():,}")
    
    # Check for data consistency issues
    print(f"\n{'='*60}")
    print("DATA CONSISTENCY CHECK")
    print(f"{'='*60}")
    
    # Check column consistency
    all_columns = set()
    for cols in summary_df['columns_list']:
        all_columns.update(cols)
    
    print(f"Unique columns across all vehicles: {len(all_columns)}")
    print("Column inconsistencies detected:")
    
    for vehicle in summary_df['vehicle']:
        vehicle_cols = set(summary_df[summary_df['vehicle'] == vehicle]['columns_list'].iloc[0])
        missing_from_others = all_columns - vehicle_cols
        if missing_from_others:
            print(f"  Vehicle {vehicle} missing: {missing_from_others}")

if __name__ == "__main__":
    main()