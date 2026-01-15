# requirements.txt
# pandas
# numpy
# polars (optional for large datasets)
# matplotlib (for visualization)

# ============================
# 1. IMPORTS AND CONFIGURATION
# ============================

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "../dataset_10EVs"  # Directory containing vehicle#.csv files
OUTPUT_DIR = "../cleaned_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vehicle metadata from README
VEHICLE_METADATA = {
    1: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 150, "sampling_hz": 0.1, "series_cells": 91},
    2: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 150, "sampling_hz": 0.1, "series_cells": 91},
    3: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 160, "sampling_hz": 0.1, "series_cells": 91},
    4: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 160, "sampling_hz": 0.1, "series_cells": 91},
    5: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 160, "sampling_hz": 0.1, "series_cells": 91},
    6: {"type": "passenger", "chemistry": "NCM", "capacity_ah": 160, "sampling_hz": 0.1, "series_cells": 91},
    7: {"type": "passenger", "chemistry": "LFP", "capacity_ah": 120, "sampling_hz": 0.5, "series_cells": None},
    8: {"type": "bus", "chemistry": "LFP", "capacity_ah": 645, "sampling_hz": 0.1, "series_cells": None},
    9: {"type": "bus", "chemistry": "LFP", "capacity_ah": 505, "sampling_hz": 0.1, "series_cells": 360},
    10: {"type": "bus", "chemistry": "LFP", "capacity_ah": 505, "sampling_hz": 0.1, "series_cells": 324},
}

# ============================
# 2. DATA LOADING FUNCTIONS
# ============================

def load_vehicle_data(vehicle_id):
    """Load a single vehicle CSV file with proper error handling."""
    file_paths = [
        Path(DATA_DIR) / f"vehicle#{vehicle_id}.xlsx",
        Path(DATA_DIR) / f"vehicle{vehicle_id}.xlsx",
        Path(DATA_DIR) / f"Vehicle#{vehicle_id}.xlsx",
    ]
    
    for fp in file_paths:
        if fp.exists():
            try:
                # Try reading with different encodings
                df = pd.read_csv(fp, encoding='utf-8')
                break
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding='latin-1')
                break
            except Exception as e:
                print(f"Error reading {fp}: {e}")
                return None
    else:
        print(f"No file found for vehicle {vehicle_id}")
        return None
    
    df['vehicle_id'] = vehicle_id
    return df

def standardize_column_names(df):
    """Standardize column names to snake_case."""
    column_mapping = {}
    for col in df.columns:
        # Skip vehicle_id
        if col == 'vehicle_id':
            continue
            
        # Standardize
        new_name = (col.lower()
                   .replace(' ', '_')
                   .replace('-', '_')
                   .replace('(', '')
                   .replace(')', ''))
        
        # Specific mappings based on observed column names
        if 'vhc_speed' in new_name or 'speed' in new_name:
            new_name = 'vhc_speed'
        elif 'charging' in new_name:
            new_name = 'charging_signal'
        elif 'totalmile' in new_name or 'total_mile' in new_name:
            new_name = 'vhc_total_mile'
        elif 'hv_voltage' in new_name or 'voltage' == new_name:
            new_name = 'hv_voltage'
        elif 'hv_current' in new_name or 'current' == new_name:
            new_name = 'hv_current'
        elif 'soc' in new_name:
            new_name = 'bcell_soc'
        elif 'maxvoltage' in new_name or 'max_voltage' in new_name:
            new_name = 'bcell_max_voltage'
        elif 'minvoltage' in new_name or 'min_voltage' in new_name:
            new_name = 'bcell_min_voltage'
        elif 'maxtemp' in new_name or 'max_temp' in new_name:
            new_name = 'bcell_max_temp'
        elif 'mintemp' in new_name or 'min_temp' in new_name:
            new_name = 'bcell_min_temp'
        elif 'temp' == new_name:
            new_name = 'bcell_temp'  # For Vehicle #7
            
        column_mapping[col] = new_name
    
    return df.rename(columns=column_mapping)

# ============================
# 3. DATA CLEANING FUNCTIONS
# ============================

def clean_numeric_string(value):
    """Convert string with comma decimal to float."""
    if pd.isna(value):
        return value
    
    if isinstance(value, str):
        # Replace comma with dot for decimal
        value = value.replace(',', '.')
        # Remove any whitespace
        value = value.strip()
        
        # Check for placeholder values
        if value in ['65535', '65535.0', 'NaN', 'NA', '']:
            return np.nan
        
        try:
            return float(value)
        except ValueError:
            return np.nan
    
    # If already numeric
    if pd.api.types.is_number(value):
        if value == 65535:
            return np.nan
        return float(value)
    
    return np.nan

def handle_special_cases(df, vehicle_id):
    """Handle vehicle-specific data issues."""
    # Vehicle #7 has different schema
    if vehicle_id == 7:
        # Check if bcell_temp exists and split to max/min if needed
        if 'bcell_temp' in df.columns and 'bcell_max_temp' not in df.columns:
            df['bcell_max_temp'] = df['bcell_temp']
            df['bcell_min_temp'] = df['bcell_temp']
            # Add missing voltage columns
            df['bcell_max_voltage'] = np.nan
            df['bcell_min_voltage'] = np.nan
    
    return df

def detect_and_clean_outliers(df, vehicle_id):
    """Detect and clean unrealistic values."""
    cleaned_df = df.copy()
    
    # 1. SOC outliers
    soc_mask = (df['bcell_soc'] < 0) | (df['bcell_soc'] > 100)
    if soc_mask.any():
        cleaned_df.loc[soc_mask, 'bcell_soc'] = np.nan
        print(f"Vehicle {vehicle_id}: Found {soc_mask.sum()} SOC outliers")
    
    # 2. Speed outliers (assuming km/h)
    speed_mask = df['vhc_speed'] > 300  # Unrealistic for EVs
    if speed_mask.any():
        cleaned_df.loc[speed_mask, 'vhc_speed'] = np.nan
        print(f"Vehicle {vehicle_id}: Found {speed_mask.sum()} speed outliers")
    
    # 3. Temperature outliers (°C)
    temp_cols = ['bcell_max_temp', 'bcell_min_temp']
    for col in temp_cols:
        if col in cleaned_df.columns:
            temp_mask = (cleaned_df[col] < -50) | (cleaned_df[col] > 100)
            if temp_mask.any():
                cleaned_df.loc[temp_mask, col] = np.nan
                print(f"Vehicle {vehicle_id}: Found {temp_mask.sum()} {col} outliers")
    
    # 4. Voltage sanity checks
    if 'hv_voltage' in cleaned_df.columns and 'bcell_max_voltage' in cleaned_df.columns:
        # Check if pack voltage aligns with cell voltages
        metadata = VEHICLE_METADATA.get(vehicle_id, {})
        series_cells = metadata.get('series_cells')
        
        if series_cells and series_cells > 0:
            # Calculate expected pack voltage from cells
            expected_min = cleaned_df['bcell_min_voltage'] * series_cells
            expected_max = cleaned_df['bcell_max_voltage'] * series_cells
            
            # Flag major mismatches (>20% difference)
            voltage_mask = (
                (cleaned_df['hv_voltage'] < 0.8 * expected_min) |
                (cleaned_df['hv_voltage'] > 1.2 * expected_max)
            ) & cleaned_df['hv_voltage'].notna()
            
            if voltage_mask.any():
                print(f"Vehicle {vehicle_id}: Found {voltage_mask.sum()} voltage mismatches")
                # Don't auto-clean, just flag
                cleaned_df['voltage_mismatch_flag'] = voltage_mask.astype(int)
    
    return cleaned_df

# ============================
# 4. MAIN PREPROCESSING PIPELINE
# ============================

def preprocess_vehicle(vehicle_id):
    """Complete preprocessing for a single vehicle."""
    print(f"\n=== Processing Vehicle {vehicle_id} ===")
    
    # 1. Load data
    df = load_vehicle_data(vehicle_id)
    if df is None:
        return None
    
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Standardize column names
    df = standardize_column_names(df)
    
    # 3. Handle special cases
    df = handle_special_cases(df, vehicle_id)
    
    # 4. Define expected columns (all vehicles should have these after standardization)
    expected_columns = [
        'vehicle_id', 'time', 'vhc_speed', 'charging_signal', 'vhc_total_mile',
        'hv_voltage', 'hv_current', 'bcell_soc', 
        'bcell_max_voltage', 'bcell_min_voltage',
        'bcell_max_temp', 'bcell_min_temp'
    ]
    
    # 5. Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns and col != 'vehicle_id':
            df[col] = np.nan
    
    # 6. Select and order columns
    df = df[expected_columns]
    
    # 7. Clean numeric values (handle comma decimals and placeholders)
    numeric_cols = [
        'vhc_speed', 'charging_signal', 'vhc_total_mile',
        'hv_voltage', 'hv_current', 'bcell_soc',
        'bcell_max_voltage', 'bcell_min_voltage',
        'bcell_max_temp', 'bcell_min_temp'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_string)
    
    # 8. Handle time column
    # Convert time to seconds if it's in milliseconds
    if df['time'].max() > 1e9:  # Likely milliseconds
        df['time_seconds'] = df['time'] / 1000
    else:
        df['time_seconds'] = df['time']
    
    # 9. Detect and clean outliers
    df = detect_and_clean_outliers(df, vehicle_id)
    
    # 10. Add metadata
    metadata = VEHICLE_METADATA.get(vehicle_id, {})
    df['vehicle_type'] = metadata.get('type', 'unknown')
    df['battery_chemistry'] = metadata.get('chemistry', 'unknown')
    df['capacity_ah'] = metadata.get('capacity_ah', np.nan)
    df['sampling_hz'] = metadata.get('sampling_hz', np.nan)
    df['series_cells'] = metadata.get('series_cells', np.nan)
    
    # 11. Add derived features
    # Voltage spread
    if 'bcell_max_voltage' in df.columns and 'bcell_min_voltage' in df.columns:
        df['cell_voltage_spread'] = df['bcell_max_voltage'] - df['bcell_min_voltage']
    
    # Temperature spread
    if 'bcell_max_temp' in df.columns and 'bcell_min_temp' in df.columns:
        df['cell_temp_spread'] = df['bcell_max_temp'] - df['bcell_min_temp']
    
    # Power (kW)
    if 'hv_voltage' in df.columns and 'hv_current' in df.columns:
        df['power_kw'] = (df['hv_voltage'] * df['hv_current']) / 1000
    
    # 12. Sort by time
    if 'time_seconds' in df.columns:
        df = df.sort_values('time_seconds')
    
    print(f"Final shape: {df.shape}")
    print(f"Missing values:")
    for col in numeric_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    return df

# ============================
# 5. MERGE ALL VEHICLES
# ============================

def create_combined_dataset():
    """Process all vehicles and combine into single dataset."""
    all_vehicles = []
    
    for vehicle_id in range(1, 11):
        vehicle_df = preprocess_vehicle(vehicle_id)
        if vehicle_df is not None:
            all_vehicles.append(vehicle_df)
            # Save individual cleaned files
            output_path = Path(OUTPUT_DIR) / f"vehicle_{vehicle_id}_cleaned.csv"
            vehicle_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
    
    if not all_vehicles:
        print("No data processed!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_vehicles, ignore_index=True)
    
    # Final sorting
    combined_df = combined_df.sort_values(['vehicle_id', 'time_seconds'])
    
    # Reset index
    combined_df = combined_df.reset_index(drop=True)
    
    # Save combined dataset
    combined_path = Path(OUTPUT_DIR) / "all_vehicles_combined.csv"
    combined_df.to_csv(combined_path, index=False)
    
    # Also save as Parquet for better compression
    parquet_path = Path(OUTPUT_DIR) / "all_vehicles_combined.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total vehicles processed: {len(all_vehicles)}")
    print(f"Total rows in combined dataset: {combined_df.shape[0]:,}")
    print(f"Total columns: {combined_df.shape[1]}")
    print(f"\nSaved files:")
    print(f"  CSV: {combined_path}")
    print(f"  Parquet: {parquet_path}")
    
    return combined_df

# ============================
# 6. QUALITY CHECK FUNCTIONS
# ============================

def run_quality_checks(df):
    """Run quality checks on the combined dataset."""
    print("\n=== QUALITY CHECKS ===")
    
    # 1. Check for missing values
    print("\n1. Missing values by column:")
    missing_stats = df.isna().sum()
    for col, missing in missing_stats.items():
        if missing > 0:
            pct = missing / len(df) * 100
            print(f"  {col}: {missing:,} ({pct:.1f}%)")
    
    # 2. Check data types
    print("\n2. Data types:")
    print(df.dtypes.to_string())
    
    # 3. Basic statistics
    print("\n3. Basic statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    print(stats_df.round(2).to_string())
    
    # 4. Check vehicle distribution
    print("\n4. Vehicle distribution:")
    vehicle_dist = df['vehicle_id'].value_counts().sort_index()
    for vid, count in vehicle_dist.items():
        pct = count / len(df) * 100
        print(f"  Vehicle {vid}: {count:,} rows ({pct:.1f}%)")
    
    # 5. Check charging states
    print("\n5. Charging signal distribution:")
    if 'charging_signal' in df.columns:
        charge_dist = df['charging_signal'].value_counts()
        for val, count in charge_dist.items():
            pct = count / len(df) * 100
            print(f"  {val}: {count:,} rows ({pct:.1f}%)")

# ============================
# 7. MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("Starting EV Dataset Preprocessing Pipeline")
    print("=" * 50)
    
    # Create sample data if files don't exist (for testing)
    if not Path(DATA_DIR).exists():
        print(f"Warning: {DATA_DIR} directory not found!")
        print("Creating sample test data...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Create a simple test file for Vehicle #1
        test_data = """time,vhc_speed,charging_signal,vhc_totalMile,hv_voltage,hv_current,bcell_soc,bcell_maxVoltage,bcell_minVoltage,bcell_maxTemp,bcell_minTemp
401042909,0,3,81491,347,4.1,61,3.831,0,21,19
401042919,0,3,81491,347,2.2,61,3.829,3.812,21,19
401042929,0,3,81491,347,3.8,61,3.829,3.812,21,19"""
        
        with open(Path(DATA_DIR) / "vehicle#1.csv", 'w') as f:
            f.write(test_data)
    
    # Run the preprocessing pipeline
    combined_data = create_combined_dataset()
    
    if combined_data is not None:
        # Run quality checks
        run_quality_checks(combined_data)
        
        # Create a data dictionary
        data_dict = {
            'vehicle_id': 'Vehicle identifier (1-10)',
            'time': 'Original timestamp (vehicle-specific format)',
            'time_seconds': 'Time in seconds (normalized)',
            'vhc_speed': 'Vehicle speed (km/h)',
            'charging_signal': '1=charging, 3=driving',
            'vhc_total_mile': 'Accumulated mileage (km)',
            'hv_voltage': 'Battery pack total voltage (V)',
            'hv_current': 'Battery pack current (A), negative=charging',
            'bcell_soc': 'State of Charge (%)',
            'bcell_max_voltage': 'Maximum cell voltage (V)',
            'bcell_min_voltage': 'Minimum cell voltage (V)',
            'bcell_max_temp': 'Maximum cell temperature (°C)',
            'bcell_min_temp': 'Minimum cell temperature (°C)',
            'cell_voltage_spread': 'Voltage difference between max and min cells',
            'cell_temp_spread': 'Temperature difference between max and min cells',
            'power_kw': 'Instantaneous power (kW)',
            'vehicle_type': 'passenger or bus',
            'battery_chemistry': 'NCM or LFP',
            'capacity_ah': 'Battery capacity in Ah',
            'sampling_hz': 'Sampling frequency',
            'series_cells': 'Number of cells in series',
        }
        
        # Save data dictionary
        dict_path = Path(OUTPUT_DIR) / "data_dictionary.csv"
        pd.DataFrame(list(data_dict.items()), columns=['column', 'description']).to_csv(dict_path, index=False)
        print(f"\nData dictionary saved: {dict_path}")
        
        print("\n=== PREPROCESSING COMPLETE ===")