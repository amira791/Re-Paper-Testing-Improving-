import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs'  # Directory containing vehicle#.xlsx files
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
    """Load a single vehicle Excel file with proper error handling."""
    file_path = Path(DATA_DIR) / f"vehicle#{vehicle_id}.xlsx"

    if not file_path.exists():
        print(f"No file found for vehicle {vehicle_id}")
        return None

    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    df['vehicle_id'] = vehicle_id
    return df


def standardize_column_names(df):
    """Standardize column names to snake_case with special handling for Vehicle 7."""
    column_mapping = {}
    
    for col in df.columns:
        # Skip vehicle_id
        if col in ['vehicle_id', 'Vehicle_id', 'VEHICLE_ID']:
            continue
        
        original_col = col
        col_lower = str(col).lower().strip()
        
        # Remove special characters and normalize
        new_name = (col_lower
                   .replace(' ', '_')
                   .replace('-', '_')
                   .replace('(', '')
                   .replace(')', '')
                   .replace('.', '')
                   .replace('__', '_')
                   .strip('_'))
        
        # Special handling for known column patterns
        if 'time' in new_name or new_name == 't':
            new_name = 'time'
        elif 'speed' in new_name:
            new_name = 'vhc_speed'
        elif 'charg' in new_name:
            new_name = 'charging_signal'
        elif any(x in new_name for x in ['mile', 'odometer', 'distance']):
            new_name = 'vhc_total_mile'
        elif any(x in new_name for x in ['hv_voltage', 'total_voltage', 'pack_voltage']):
            new_name = 'hv_voltage'
        elif any(x in new_name for x in ['hv_current', 'total_current', 'pack_current']):
            new_name = 'hv_current'
        elif 'soc' in new_name:
            new_name = 'bcell_soc'
        elif any(x in new_name for x in ['maxvoltage', 'max_v', 'max_cell_v']):
            new_name = 'bcell_max_voltage'
        elif any(x in new_name for x in ['minvoltage', 'min_v', 'min_cell_v']):
            new_name = 'bcell_min_voltage'
        elif any(x in new_name for x in ['maxtemp', 'max_t', 'max_cell_temp']):
            new_name = 'bcell_max_temp'
        elif any(x in new_name for x in ['mintemp', 'min_t', 'min_cell_temp']):
            new_name = 'bcell_min_temp'
        elif 'temp' == new_name or 'temperature' in new_name:
            # Special case for Vehicle 7: might be single temperature column
            new_name = 'bcell_temp'
        
        # Fix inconsistent naming from your sample data
        if new_name == 'vhc_totalmile':
            new_name = 'vhc_total_mile'
        elif new_name == 'bcell_maxvoltage':
            new_name = 'bcell_max_voltage'
        elif new_name == 'bcell_minvoltage':
            new_name = 'bcell_min_voltage'
        elif new_name == 'bcell_maxtemp':
            new_name = 'bcell_max_temp'
        elif new_name == 'bcell_mintemp':
            new_name = 'bcell_min_temp'
        
        column_mapping[original_col] = new_name
        print(f"  Mapping: '{original_col}' → '{new_name}'")
    
    return df.rename(columns=column_mapping)

def handle_vehicle7_special_case(df, vehicle_id):
    """Special handling for Vehicle 7 which has different column structure."""
    if vehicle_id != 7:
        return df
    
    print("Applying Vehicle 7 special handling...")
    
    # Check what columns we have after standardization
    available_cols = df.columns.tolist()
    print(f"Available columns: {available_cols}")
    
    # If Vehicle 7 has only 'bcell_temp' (not min/max), create min/max columns
    if 'bcell_temp' in available_cols and 'bcell_max_temp' not in available_cols:
        print("  Creating bcell_max_temp and bcell_min_temp from bcell_temp")
        df['bcell_max_temp'] = df['bcell_temp']
        df['bcell_min_temp'] = df['bcell_temp']
    
    # Vehicle 7 might have different charging signal values
    if 'charging_signal' in available_cols:
        # Check what values exist
        unique_signals = df['charging_signal'].unique()
        print(f"  Unique charging signals: {sorted(unique_signals)}")
        
        # Save original for reference
        df['charging_signal_original'] = df['charging_signal']
        
        # Try to map based on patterns
        # This will need adjustment after seeing actual data
        if set(unique_signals) <= {0, 1, 3}:
            # Might already be using standard coding
            pass
        else:
            print(f"  Warning: Unexpected charging signal values in Vehicle 7")
    
    # Check for missing voltage columns
    voltage_needed = ['bcell_max_voltage', 'bcell_min_voltage']
    missing_voltage = [col for col in voltage_needed if col not in available_cols]
    if missing_voltage:
        print(f"  Warning: Missing voltage columns: {missing_voltage}")
    
    return df

def standardize_data_types(df, vehicle_id):
    """Ensure consistent data types across all vehicles."""
    
    # Get metadata for this vehicle
    metadata = VEHICLE_METADATA.get(vehicle_id, {})
    
    # Define expected numeric columns
    numeric_columns = [
        'time', 'vhc_speed', 'charging_signal', 'vhc_total_mile',
        'hv_voltage', 'hv_current', 'bcell_soc', 
        'bcell_max_voltage', 'bcell_min_voltage',
        'bcell_max_temp', 'bcell_min_temp', 'bcell_temp'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, handling European decimal commas
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Add metadata columns
    for key, value in metadata.items():
        df[f'metadata_{key}'] = value
    
    return df

def load_and_standardize_all_vehicles():
    """Load and standardize all 10 vehicle datasets."""
    all_vehicles = {}
    
    print("=" * 60)
    print("LOADING AND STANDARDIZING VEHICLE DATA")
    print("=" * 60)
    
    for vehicle_id in range(1, 11):
        print(f"\nProcessing Vehicle {vehicle_id}...")
        
        # Load the data
        df = load_vehicle_data(vehicle_id)
        if df is None:
            print(f"  Skipping Vehicle {vehicle_id} - could not load data")
            continue
        
        # Standardize column names
        print("  Standardizing column names...")
        df = standardize_column_names(df)
        
        # Special handling for Vehicle 7
        if vehicle_id == 7:
            df = handle_vehicle7_special_case(df, vehicle_id)
        
        # Standardize data types
        print("  Standardizing data types...")
        df = standardize_data_types(df, vehicle_id)
        
        # Display column info
        print(f"  Final columns: {df.columns.tolist()}")
        print(f"  Data shape: {df.shape}")
        
        # Store in dictionary
        all_vehicles[vehicle_id] = df
        
        # Save to output directory
        output_file = Path(OUTPUT_DIR) / f"vehicle_{vehicle_id}_standardized.parquet"
        df.to_parquet(output_file, index=False)
        print(f"  Saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print(f"Successfully loaded {len(all_vehicles)} vehicles")
    print("=" * 60)
    
    return all_vehicles

# ============================
# 3. TESTING FUNCTION
# ============================

def test_loading_functions():
    """Test the loading and standardization functions."""
    
    print("Testing loading functions...")
    
    # Test 1: Load a single vehicle
    print("\n1. Testing load_vehicle_data()...")
    test_vehicle_id = 1
    df_test = load_vehicle_data(test_vehicle_id)
    
    if df_test is not None:
        print(f"   Successfully loaded Vehicle {test_vehicle_id}")
        print(f"   Shape: {df_test.shape}")
        print(f"   Columns: {list(df_test.columns)[:10]}...")  # First 10 columns
    else:
        print(f"   Failed to load Vehicle {test_vehicle_id}")
        # Try another vehicle
        test_vehicle_id = 7
        df_test = load_vehicle_data(test_vehicle_id)
        if df_test is not None:
            print(f"   Successfully loaded Vehicle {test_vehicle_id} as fallback")
            print(f"   Shape: {df_test.shape}")
            print(f"   Columns: {list(df_test.columns)}")
    
    # Test 2: Standardize column names
    if df_test is not None:
        print("\n2. Testing standardize_column_names()...")
        df_standardized = standardize_column_names(df_test.copy())
        print(f"   Original columns: {list(df_test.columns)[:10]}...")
        print(f"   Standardized columns: {list(df_standardized.columns)[:10]}...")
    
    # Test 3: Load all vehicles
    print("\n3. Testing load_and_standardize_all_vehicles()...")
    all_vehicles = load_and_standardize_all_vehicles()
    
    # Display summary
    if all_vehicles:
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        summary_data = []
        for vehicle_id, df in all_vehicles.items():
            summary_data.append({
                'Vehicle': vehicle_id,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Chemistry': VEHICLE_METADATA[vehicle_id]['chemistry'],
                'Type': VEHICLE_METADATA[vehicle_id]['type'],
                'Missing Values': df.isnull().sum().sum()
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Check for common columns
        common_columns = set.intersection(*[set(df.columns) for df in all_vehicles.values()])
        print(f"\nCommon columns across all vehicles ({len(common_columns)}):")
        print(sorted(common_columns))
        
        # Identify Vehicle 7 differences
        if 7 in all_vehicles:
            print(f"\nVehicle 7 unique columns:")
            vehicle7_cols = set(all_vehicles[7].columns)
            other_cols = set.union(*[set(all_vehicles[v].columns) for v in all_vehicles if v != 7])
            unique_to_7 = vehicle7_cols - other_cols
            print(f"  Unique: {sorted(unique_to_7)}")
            
            missing_in_7 = other_cols - vehicle7_cols
            print(f"  Missing in 7: {sorted(missing_in_7)}")
    
    return all_vehicles

# ============================
# 4. MAIN EXECUTION
# ============================

if __name__ == "__main__":
    # Run the tests
    all_vehicles = test_loading_functions()
    
    # If you want to work with the loaded data in interactive mode
    if all_vehicles:
        print("\n" + "=" * 60)
        print("DATA IS READY FOR ANALYSIS")
        print("=" * 60)
        print("\nAccess individual vehicles via all_vehicles[vehicle_id]")
        print("Example: df_vehicle1 = all_vehicles[1]")
        print("\nNext steps:")
        print("1. Check data quality with df.isnull().sum()")
        print("2. Explore with df.describe()")
        print("3. Look for the 65535 error values")
        print("4. Investigate time column format")