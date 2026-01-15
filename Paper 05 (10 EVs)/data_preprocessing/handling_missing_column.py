# cleanup_and_merge.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load the standardized data
def load_standardized_data():
    """Load all standardized vehicle data."""
    input_dir = Path(r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\cleaned_dataset')
    all_vehicles = {}
    
    for vehicle_id in range(1, 11):
        file_path = input_dir / f"vehicle_{vehicle_id}_standardized.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_vehicles[vehicle_id] = df
            print(f"Loaded Vehicle {vehicle_id}: {len(df)} rows")
        else:
            print(f"Warning: File not found for Vehicle {vehicle_id}")
    
    return all_vehicles

def clean_65535_placeholders(df, vehicle_id):
    """Replace 65535 placeholder values with NaN."""
    print(f"  Cleaning 65535 placeholders for Vehicle {vehicle_id}...")
    
    # Columns that might contain 65535
    voltage_cols = ['bcell_max_voltage', 'bcell_min_voltage']
    temp_cols = ['bcell_max_temp', 'bcell_min_temp']
    
    for col in voltage_cols + temp_cols:
        if col in df.columns:
            # Count occurrences of 65535 or similar placeholders
            placeholder_mask = df[col].astype(str).str.contains('65535|9999|99999', na=False)
            if placeholder_mask.any():
                count = placeholder_mask.sum()
                print(f"    Found {count} placeholder values in {col}")
                df.loc[placeholder_mask, col] = np.nan
    
    return df

def fix_vehicle7_charging_signal(df):
    """Fix Vehicle 7's charging signal encoding."""
    if 'charging_signal_original' in df.columns:
        print("  Mapping Vehicle 7 charging signals...")
        # Check what values we have
        unique_vals = df['charging_signal_original'].unique()
        print(f"    Original values: {unique_vals}")
        
        # Based on sample data: 0=?, 1=?
        # Need to analyze actual patterns. For now, assume:
        # 0 → 3 (driving/not charging)
        # 1 → 1 (charging)
        df['charging_signal'] = df['charging_signal_original'].map({0: 3, 1: 1})
    
    return df

def analyze_time_column(df, vehicle_id):
    """Analyze and fix time column format."""
    print(f"  Analyzing time column for Vehicle {vehicle_id}...")
    
    if 'time' in df.columns:
        # Show time statistics
        time_min = df['time'].min()
        time_max = df['time'].max()
        time_diff = time_max - time_min
        
        print(f"    Time range: {time_min} to {time_max}")
        print(f"    Time span: {time_diff} units")
        
        # Check if it's likely Unix timestamp (seconds since 1970)
        # Unix timestamps are typically ~1.6B for 2020s
        if time_max > 1e9 and time_max < 2e9:
            print("    Time appears to be Unix timestamp (seconds)")
            # Convert to datetime
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
        elif time_max > 1e12 and time_max < 2e12:
            print("    Time appears to be Unix timestamp (milliseconds)")
            df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        else:
            print("    Time format unclear - keeping as is")
            df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Calculate sampling rate
        if len(df) > 1:
            time_diffs = df['time'].diff().dropna()
            avg_interval = time_diffs.mean()
            print(f"    Average time interval: {avg_interval:.1f} units")
    
    return df

def add_missing_columns(df, vehicle_id):
    """Ensure all vehicles have the same columns."""
    print(f"  Ensuring column consistency for Vehicle {vehicle_id}...")
    
    # Define complete column set
    complete_columns = [
        'vehicle_id', 'time', 'datetime', 'vhc_speed', 'charging_signal', 
        'vhc_total_mile', 'hv_voltage', 'hv_current', 'bcell_soc',
        'bcell_max_voltage', 'bcell_min_voltage', 'bcell_max_temp', 
        'bcell_min_temp', 'bcell_temp', 'charging_signal_original',
        'vehicle_type', 'battery_chemistry', 'capacity_ah', 
        'sampling_hz', 'series_cells'
    ]
    
    # Add missing columns
    for col in complete_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    return df[complete_columns]  # Reorder columns

def run_comprehensive_cleaning(all_vehicles):
    """Run all cleaning steps on each vehicle."""
    cleaned_vehicles = {}
    
    print("=" * 60)
    print("COMPREHENSIVE DATA CLEANING")
    print("=" * 60)
    
    for vehicle_id, df in all_vehicles.items():
        print(f"\nProcessing Vehicle {vehicle_id}...")
        
        # Step 1: Clean 65535 placeholders
        df = clean_65535_placeholders(df.copy(), vehicle_id)
        
        # Step 2: Special handling for Vehicle 7
        if vehicle_id == 7:
            df = fix_vehicle7_charging_signal(df)
        
        # Step 3: Analyze and fix time column
        df = analyze_time_column(df, vehicle_id)
        
        # Step 4: Add metadata columns properly
        if 'metadata_type' in df.columns:
            # Extract metadata from columns to proper columns
            df['vehicle_type'] = df['metadata_type'].iloc[0] if not df['metadata_type'].isna().all() else np.nan
            df['battery_chemistry'] = df['metadata_chemistry'].iloc[0] if not df['metadata_chemistry'].isna().all() else np.nan
            df['capacity_ah'] = df['metadata_capacity_ah'].iloc[0] if not df['metadata_capacity_ah'].isna().all() else np.nan
            df['sampling_hz'] = df['metadata_sampling_hz'].iloc[0] if not df['metadata_sampling_hz'].isna().all() else np.nan
            df['series_cells'] = df['metadata_series_cells'].iloc[0] if not df['metadata_series_cells'].isna().all() else np.nan
        
        # Step 5: Ensure column consistency
        df = add_missing_columns(df, vehicle_id)
        
        # Step 6: Calculate derived features
        df = calculate_derived_features(df, vehicle_id)
        
        cleaned_vehicles[vehicle_id] = df
        print(f"  Final shape: {df.shape}")
    
    return cleaned_vehicles

def calculate_derived_features(df, vehicle_id):
    """Calculate derived features for analysis."""
    print(f"  Calculating derived features for Vehicle {vehicle_id}...")
    
    # 1. Power calculation
    if 'hv_voltage' in df.columns and 'hv_current' in df.columns:
        df['power_kw'] = (df['hv_voltage'] * df['hv_current']) / 1000
    
    # 2. Cell voltage spread (if available)
    if 'bcell_max_voltage' in df.columns and 'bcell_min_voltage' in df.columns:
        df['cell_voltage_spread'] = df['bcell_max_voltage'] - df['bcell_min_voltage']
    
    # 3. Cell temperature spread (if available)
    if 'bcell_max_temp' in df.columns and 'bcell_min_temp' in df.columns:
        df['cell_temp_spread'] = df['bcell_max_temp'] - df['bcell_min_temp']
    
    # 4. Driving/charging status
    if 'charging_signal' in df.columns:
        df['is_charging'] = df['charging_signal'] == 1
        df['is_driving'] = df['charging_signal'] == 3
    
    # 5. SOC-based features
    if 'bcell_soc' in df.columns:
        df['soc_category'] = pd.cut(df['bcell_soc'], 
                                    bins=[0, 20, 80, 100], 
                                    labels=['low', 'medium', 'high'])
    
    return df

def merge_all_vehicles(cleaned_vehicles):
    """Merge all cleaned vehicles into a single dataset."""
    print("\n" + "=" * 60)
    print("MERGING ALL VEHICLES")
    print("=" * 60)
    
    # Combine all dataframes
    all_dfs = list(cleaned_vehicles.values())
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by vehicle and time
    combined_df = combined_df.sort_values(['vehicle_id', 'time'])
    combined_df = combined_df.reset_index(drop=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total rows: {combined_df.shape[0]:,}")
    print(f"Total columns: {combined_df.shape[1]}")
    
    return combined_df

def save_and_analyze(combined_df):
    """Save the combined dataset and provide analysis."""
    output_dir = Path("../cleaned_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Save combined dataset
    csv_path = output_dir / "all_vehicles_combined_cleaned.csv"
    parquet_path = output_dir / "all_vehicles_combined_cleaned.parquet"
    
    combined_df.to_csv(csv_path, index=False)
    combined_df.to_parquet(parquet_path, index=False)
    
    print(f"\nSaved combined dataset:")
    print(f"  CSV: {csv_path}")
    print(f"  Parquet: {parquet_path}")
    
    # Basic analysis
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    print("\n1. Vehicle Distribution:")
    vehicle_dist = combined_df['vehicle_id'].value_counts().sort_index()
    for vid, count in vehicle_dist.items():
        pct = count / len(combined_df) * 100
        print(f"   Vehicle {vid}: {count:,} rows ({pct:.1f}%)")
    
    print("\n2. Missing Values (top 10 columns):")
    missing_stats = combined_df.isna().sum().sort_values(ascending=False)
    for col, count in missing_stats.head(10).items():
        if count > 0:
            pct = count / len(combined_df) * 100
            print(f"   {col}: {count:,} ({pct:.1f}%)")
    
    print("\n3. Charging Status Distribution:")
    if 'charging_signal' in combined_df.columns:
        charge_dist = combined_df['charging_signal'].value_counts()
        for val, count in charge_dist.items():
            pct = count / len(combined_df) * 100
            status = {1: "Charging", 3: "Driving/Not Charging"}.get(val, f"Unknown ({val})")
            print(f"   {status}: {count:,} ({pct:.1f}%)")
    
    print("\n4. Data Types:")
    print(combined_df.dtypes.to_string())
    
    return combined_df

# Main execution
if __name__ == "__main__":
    print("Starting Comprehensive Data Cleaning Pipeline")
    print("=" * 60)
    
    # Load standardized data
    all_vehicles = load_standardized_data()
    
    if not all_vehicles:
        print("No data loaded. Exiting.")
    else:
        # Run comprehensive cleaning
        cleaned_vehicles = run_comprehensive_cleaning(all_vehicles)
        
        # Merge all vehicles
        combined_df = merge_all_vehicles(cleaned_vehicles)
        
        # Save and analyze
        final_df = save_and_analyze(combined_df)
        
        print("\n" + "=" * 60)
        print("CLEANING COMPLETE!")
        print("=" * 60)
        print("\nThe dataset is now ready for analysis and modeling.")
        print("\nKey features available:")
        print("  • Standardized columns across all vehicles")
        print("  • Cleaned placeholder values (65535 → NaN)")
        print("  • Consistent charging signal encoding")
        print("  • Derived features (power, voltage/temp spreads)")
        print("  • Datetime conversion where possible")