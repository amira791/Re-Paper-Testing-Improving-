"""
Basic Statistics Analysis
Analyze dataset structure, missing values, and basic statistics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load the combined dataset"""
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    df = pd.read_parquet(data_path / "all_vehicles_combined.parquet")
    return df

def analyze_dataset_structure(df):
    """Analyze basic structure of the dataset"""
    print("="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Basic info
    print(f"\n1. Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Column types
    print("\n2. Column Data Types:")
    dtypes_summary = df.dtypes.value_counts()
    for dtype, count in dtypes_summary.items():
        print(f"   {dtype}: {count} columns")
    
    # Memory usage
    print(f"\n3. Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    # Missing values by column
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percentage': missing_percentage
    })
    
    # Filter columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print("\nColumns with Missing Values:")
        print(missing_df.to_string())
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        bars = plt.bar(missing_df.index[:20], missing_df['Missing_Percentage'][:20])
        plt.title('Missing Values Percentage by Column (Top 20)', fontsize=14)
        plt.xlabel('Columns', fontsize=12)
        plt.ylabel('Missing Values (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("\n✅ No missing values found in the dataset!")
    
    return missing_df

def analyze_basic_statistics(df):
    """Calculate basic statistics for numerical columns"""
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nNumerical Columns ({len(numerical_cols)}):")
    for i, col in enumerate(numerical_cols, 1):
        print(f"{i:3}. {col}")
    
    # Calculate statistics for key columns
    key_columns = ['vhc_speed', 'hv_voltage', 'hv_current', 'bcell_soc', 
                   'bcell_maxVoltage', 'bcell_minVoltage', 'bcell_maxTemp', 
                   'bcell_minTemp', 'soh_capacity', 'vhc_totalMile']
    
    available_cols = [col for col in key_columns if col in df.columns]
    
    stats_df = pd.DataFrame()
    for col in available_cols:
        stats_df[col] = [
            df[col].min(),
            df[col].max(),
            df[col].mean(),
            df[col].median(),
            df[col].std(),
            df[col].skew(),
            df[col].kurtosis()
        ]
    
    stats_df.index = ['Min', 'Max', 'Mean', 'Median', 'Std', 'Skewness', 'Kurtosis']
    
    print("\nStatistics for Key Numerical Columns:")
    print(stats_df.T.round(3).to_string())
    
    return stats_df

def analyze_categorical_variables(df):
    """Analyze categorical variables"""
    print("\n" + "="*80)
    print("CATEGORICAL VARIABLES ANALYSIS")
    print("="*80)
    
    # Check for categorical columns
    categorical_cols = ['vehicle_id', 'chemistry', 'vehicle_type', 'charging_signal']
    available_cats = [col for col in categorical_cols if col in df.columns]
    
    for col in available_cats:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        
        summary = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages.round(2)
        })
        print(summary.to_string())
        
        # Visualization for top categories
        plt.figure(figsize=(10, 5))
        
        if len(value_counts) <= 10:  # For small number of categories
            bars = plt.bar(value_counts.index.astype(str), value_counts.values)
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                        f'{height:,}', ha='center', va='bottom', fontsize=9)
        else:  # For many categories (like vehicle_id)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(range(len(value_counts)), value_counts.index.astype(str), rotation=45)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'categorical_{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return available_cats

def analyze_time_series_structure(df):
    """Analyze time series structure if time column exists"""
    print("\n" + "="*80)
    print("TIME SERIES STRUCTURE")
    print("="*80)
    
    if 'time' in df.columns:
        print(f"\nTime column found!")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"Time unique values: {df['time'].nunique():,}")
        
        # Check for time gaps
        if df['time'].dtype in [np.int64, np.float64]:
            time_diff = df['time'].diff().dropna()
            print(f"\nTime interval statistics:")
            print(f"  Mean interval: {time_diff.mean():.2f}")
            print(f"  Std interval: {time_diff.std():.2f}")
            print(f"  Min interval: {time_diff.min():.2f}")
            print(f"  Max interval: {time_diff.max():.2f}")
            
            # Check for regular sampling
            unique_intervals = time_diff.round(2).unique()
            print(f"  Unique intervals: {len(unique_intervals)}")
            if len(unique_intervals) <= 5:
                print(f"  Interval values: {sorted(unique_intervals)}")
    
    return 'time' in df.columns

def generate_summary_report(df, missing_df, stats_df, has_time_series):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    summary = {
        'Total Rows': f"{len(df):,}",
        'Total Columns': len(df.columns),
        'Numerical Columns': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical Columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'Missing Columns': len(missing_df) if len(missing_df) > 0 else 0,
        'Has Time Series': "Yes" if has_time_series else "No",
        'Unique Vehicles': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else "N/A",
        'Chemistry Types': df['chemistry'].unique().tolist() if 'chemistry' in df.columns else "N/A",
        'Vehicle Types': df['vehicle_type'].unique().tolist() if 'vehicle_type' in df.columns else "N/A",
        'SOH Range': f"{df['soh_capacity'].min():.1f}% - {df['soh_capacity'].max():.1f}%" if 'soh_capacity' in df.columns else "N/A",
        'Average SOH': f"{df['soh_capacity'].mean():.1f}%" if 'soh_capacity' in df.columns else "N/A"
    }
    
    print("\n📊 DATASET SUMMARY:")
    for key, value in summary.items():
        print(f"{key:25}: {value}")
    
    # Save summary to file
    with open('dataset_summary.txt', 'w') as f:
        f.write("DATASET SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key:25}: {value}\n")
    
    print("\n✅ Summary saved to 'dataset_summary.txt'")

def main():
    """Main analysis function"""
    print("Starting Basic Statistics Analysis...")
    
    # Load data
    df = load_data()
    
    # Perform analyses
    df = analyze_dataset_structure(df)
    missing_df = analyze_missing_values(df)
    stats_df = analyze_basic_statistics(df)
    analyze_categorical_variables(df)
    has_time_series = analyze_time_series_structure(df)
    
    # Generate summary report
    generate_summary_report(df, missing_df, stats_df, has_time_series)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("1. missing_values_analysis.png")
    print("2. categorical_*_distribution.png (multiple files)")
    print("3. dataset_summary.txt")

if __name__ == "__main__":
    main()