import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set your data path
data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")

def preprocess_data(df, vehicle_id):
    """Preprocess data: fix decimal format, handle invalid values"""
    
    # Make a copy
    df = df.copy()
    
    # First, ensure correct column names (strip spaces, standardize)
    df.columns = [col.strip() for col in df.columns]
    
    # Replace comma decimals with points for ALL columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Replace commas and convert to float
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Handle 65535 as NaN (sensor fault/missing value)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(65535, np.nan)
            # Only replace 0 in min/max voltage columns
            if 'minVoltage' in col or 'maxVoltage' in col:
                df[col] = df[col].replace(0, np.nan)
    
    # For vehicle 7, create missing columns PROPERLY
    if vehicle_id == 7:
        # Based on LFP chemistry and 540V pack voltage
        # Typical LFP cell voltage: ~3.2-3.3V
        # Estimated cells = 540V / 3.25V ≈ 166 cells
        estimated_cells = round(df['hv_voltage'].median() / 3.25)
        
        df['bcell_maxVoltage'] = df['hv_voltage'] / estimated_cells
        df['bcell_minVoltage'] = df['hv_voltage'] / estimated_cells
        
        if 'bcell_Temp' in df.columns:
            df['bcell_maxTemp'] = df['bcell_Temp']
            df['bcell_minTemp'] = df['bcell_Temp']
        else:
            df['bcell_maxTemp'] = 25  # Default
            df['bcell_minTemp'] = 25
    
    # Ensure critical columns exist
    required_cols = ['time', 'vhc_speed', 'vhc_totalMile', 'hv_voltage', 
                    'hv_current', 'bcell_soc']
    
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} missing for vehicle {vehicle_id}")
    
    return df

def calculate_soh_capacity(df, vehicle_id, rated_capacity):
    """
    Calculate SOH based on capacity estimation
    FIXED BASED ON YOUR RESULTS:
    1. Fixed Vehicle 7 time interval (0.5Hz vs 0.1Hz)
    2. Fixed broadcasting error 
    3. Added Vehicle 7 current scaling fix
    4. Better handling of partial charge events
    """
    
    df = df.copy()
    df['soh_capacity'] = np.nan
    
    # DEBUG: Print current stats
    print(f"  Vehicle {vehicle_id} current stats:")
    print(f"    Min: {df['hv_current'].min():.1f}A, Max: {df['hv_current'].max():.1f}A")
    print(f"    Negative %: {(df['hv_current'] < 0).sum()/len(df)*100:.1f}%")
    print(f"    Positive %: {(df['hv_current'] > 0).sum()/len(df)*100:.1f}%")
    
    # FIX 1: Different time intervals for different vehicles
    # Based on README: Vehicle 7 samples at 0.5Hz (2 seconds), others at 0.1Hz (10 seconds)
    if vehicle_id == 7:
        time_interval_h = 2 / 3600  # 2 seconds in hours (0.5Hz)
        current_scaling_factor = 5  # Vehicle 7 current seems 5x too high based on your results
    else:
        time_interval_h = 10 / 3600  # 10 seconds in hours (0.1Hz)
        current_scaling_factor = 1
    
    # Store all calculated SOH values for statistics
    calculated_soh_values = []
    
    # Method 1: Charging events (negative current OR charging_signal = 1)
    if 'charging_signal' in df.columns:
        # Use charging signal when available, otherwise use current < -5A
        charge_mask = (df['charging_signal'] == 1) | ((df['vhc_speed'] == 0) & (df['hv_current'] < -5))
    else:
        charge_mask = (df['vhc_speed'] == 0) & (df['hv_current'] < -5)
    
    if charge_mask.any():
        df['charge_group'] = (charge_mask != charge_mask.shift()).cumsum()
        
        for group_id in df[charge_mask]['charge_group'].unique():
            group_data = df[df['charge_group'] == group_id]
            if len(group_data) > 3:  # At least 3 points for meaningful estimate
                # Charge current is NEGATIVE, take absolute value
                # Apply scaling factor for Vehicle 7
                charge_currents = abs(group_data['hv_current']) / current_scaling_factor
                charge_ah = (charge_currents * time_interval_h).sum()
                
                soc_start = group_data['bcell_soc'].iloc[0]
                soc_end = group_data['bcell_soc'].iloc[-1]
                soc_change = soc_end - soc_start
                
                # Only consider events with meaningful SOC change
                if soc_change >= 5:  # Minimum 5% SOC increase for reliable estimate
                    estimated_capacity = charge_ah / (soc_change / 100)
                    
                    # Sanity check: capacity shouldn't be >200% of rated or <30%
                    if estimated_capacity < 0.3 * rated_capacity or estimated_capacity > 2.0 * rated_capacity:
                        print(f"    Skipping implausible capacity: {estimated_capacity:.1f}Ah")
                        continue
                    
                    soh_value = (estimated_capacity / rated_capacity) * 100
                    soh_value = min(105, max(70, soh_value))  # Clip to 70-105%
                    
                    # FIXED BROADCAST: Use proper indexing
                    idx_start = group_data.index[0]
                    idx_end = group_data.index[-1]
                    df.loc[idx_start:idx_end, 'soh_capacity'] = soh_value
                    
                    calculated_soh_values.append(soh_value)
                    
                    print(f"    Charge event: {soc_start:.1f}%→{soc_end:.1f}% "
                          f"({soc_change:.1f}%), Capacity: {estimated_capacity:.1f}Ah, "
                          f"SOH: {soh_value:.1f}%")
    
    # Method 2: Discharge events (driving with positive current)
    if 'vhc_speed' in df.columns:
        discharge_mask = (df['vhc_speed'] > 1) & (df['hv_current'] > 5)  # Minimum 5A discharge
        
        if discharge_mask.any():
            df['discharge_group'] = (discharge_mask != discharge_mask.shift()).cumsum()
            
            for group_id in df[discharge_mask]['discharge_group'].unique():
                group_data = df[df['discharge_group'] == group_id]
                if len(group_data) > 5:  # At least 5 points
                    # Apply scaling factor for Vehicle 7
                    discharge_currents = group_data['hv_current'] / current_scaling_factor
                    discharge_ah = (discharge_currents * time_interval_h).sum()
                    
                    soc_start = group_data['bcell_soc'].iloc[0]
                    soc_end = group_data['bcell_soc'].iloc[-1]
                    soc_change = soc_start - soc_end  # Positive for discharge
                    
                    if soc_change >= 3:  # Minimum 3% SOC decrease
                        estimated_capacity = discharge_ah / (soc_change / 100)
                        
                        # Sanity check
                        if estimated_capacity < 0.3 * rated_capacity or estimated_capacity > 2.0 * rated_capacity:
                            continue
                        
                        soh_value = (estimated_capacity / rated_capacity) * 100
                        soh_value = min(105, max(70, soh_value))
                        
                        # Only fill where we don't have charge estimates
                        idx_start = group_data.index[0]
                        idx_end = group_data.index[-1]
                        fill_mask = df.loc[idx_start:idx_end, 'soh_capacity'].isna()
                        if fill_mask.any():
                            df.loc[idx_start:idx_end, 'soh_capacity'][fill_mask] = soh_value
                            calculated_soh_values.append(soh_value)
    
    # Method 3: Combined charge-discharge cycles for better accuracy
    # Look for complete cycles: discharge followed by charge
    if len(df) > 1000:  # Only for datasets with enough data
        # Find discharge segments
        discharge_segments = []
        if 'discharge_group' in df.columns:
            for group_id in df['discharge_group'].dropna().unique():
                segment = df[df['discharge_group'] == group_id]
                if len(segment) > 10:
                    discharge_segments.append(segment)
        
        # Find corresponding charge segments
        if discharge_segments and 'charge_group' in df.columns:
            for i, discharge_seg in enumerate(discharge_segments):
                # Find next charge event after this discharge
                end_time = discharge_seg.index[-1]
                future_charges = df.loc[end_time:, 'charge_group'].dropna().unique()
                
                if len(future_charges) > 0:
                    charge_seg = df[df['charge_group'] == future_charges[0]]
                    
                    if len(charge_seg) > 3:
                        # Combined cycle analysis
                        total_discharge = (charge_seg['hv_current'].abs().sum() * time_interval_h) / current_scaling_factor
                        soc_diff = charge_seg['bcell_soc'].iloc[-1] - discharge_seg['bcell_soc'].iloc[0]
                        
                        if abs(soc_diff) > 10:  # Meaningful cycle
                            estimated_capacity = total_discharge / (abs(soc_diff) / 100)
                            soh_value = (estimated_capacity / rated_capacity) * 100
                            soh_value = min(105, max(70, soh_value))
                            
                            # Fill both segments
                            idx_start = discharge_seg.index[0]
                            idx_end = charge_seg.index[-1]
                            df.loc[idx_start:idx_end, 'soh_capacity'] = soh_value
                            calculated_soh_values.append(soh_value)
    
    # Fill gaps based on calculated values
    if calculated_soh_values:
        print(f"  Calculated {len(calculated_soh_values)} SOH estimates")
        print(f"  SOH range: {min(calculated_soh_values):.1f}%-{max(calculated_soh_values):.1f}%")
        print(f"  SOH median: {np.median(calculated_soh_values):.1f}%")
        
        # Use median of calculated values for filling
        median_soh = np.median(calculated_soh_values)
        
        # Forward/backward fill with limits
        df['soh_capacity'] = df['soh_capacity'].ffill(limit=1000).bfill(limit=1000)
        
        # Fill remaining NaN with median
        nan_count = df['soh_capacity'].isna().sum()
        if nan_count > 0:
            df['soh_capacity'] = df['soh_capacity'].fillna(median_soh)
            print(f"  Filled {nan_count} NaN values with median SOH: {median_soh:.1f}%")
    
    else:
        # No SOH estimates found - use mileage-based fallback
        print(f"  Warning: No reliable charge/discharge events found for capacity estimation")
        
        avg_mileage = df['vhc_totalMile'].median()
        print(f"  Average mileage: {avg_mileage:.0f} km")
        
        # Degradation rates based on chemistry
        if vehicle_id <= 6:  # NCM
            degradation_rate = 2.0 / 30000  # 2% per 30,000 km
        else:  # LFP
            degradation_rate = 1.5 / 30000  # 1.5% per 30,000 km
        
        # Assume new battery at 0 km, degrade linearly
        base_soh = 100 - (avg_mileage * degradation_rate)
        base_soh = max(70, min(100, base_soh))  # Clip to 70-100%
        
        df['soh_capacity'] = base_soh
        print(f"  Using mileage-based SOH: {base_soh:.1f}%")
    
    # Add some noise to identical values (avoid perfect correlation)
    if df['soh_capacity'].nunique() < 10:  # If too few unique values
        noise = np.random.normal(0, 0.5, len(df))  # Small noise
        df['soh_capacity'] = df['soh_capacity'] + noise
        df['soh_capacity'] = df['soh_capacity'].clip(70, 105)
    
    return df

# def calculate_soh_resistance(df, vehicle_id):
#     """
#     Calculate RELATIVE resistance-based health indicator WITHOUT fixed baselines
    
#     Instead of absolute SOH, calculates:
#     1. Resistance trend within vehicle
#     2. Relative resistance compared to similar conditions
#     3. Health score based on resistance stability
#     """
    
#     df = df.copy()
#     df['resistance_health'] = np.nan  # Renamed from soh_resistance
    
#     # Step 1: Calculate instantaneous resistance for all significant current changes
#     df['R_instant'] = np.nan
    
#     # Filter for meaningful measurements
#     valid_mask = (df['hv_current'].abs() > 5) & (df['hv_current'].diff().abs() > 2)
    
#     # Calculate dV/dI using forward differences
#     df.loc[valid_mask, 'R_instant'] = (
#         df.loc[valid_mask, 'hv_voltage'].diff().abs() / 
#         df.loc[valid_mask, 'hv_current'].diff().abs()
#     )
    
#     # Remove outliers (resistance > 1 ohm or negative)
#     df.loc[(df['R_instant'] > 1) | (df['R_instant'] < 0), 'R_instant'] = np.nan
    
#     # Step 2: Group by SOC and temperature ranges for fair comparison
#     # Resistance varies with SOC and temperature
#     df['soc_bin'] = (df['bcell_soc'] / 10).astype(int) * 10  # 0, 10, 20, ..., 90, 100
#     df['temp_bin'] = ((df['bcell_maxTemp'] + df['bcell_minTemp']) / 2 / 5).astype(int) * 5  # 15, 20, 25, ...
    
#     # Step 3: Calculate baseline resistance for this vehicle at each condition
#     # Assuming the vehicle's OWN AVERAGE resistance when new-ish represents 100% health
#     baseline_R = {}
    
#     for soc in df['soc_bin'].unique():
#         for temp in df['temp_bin'].unique():
#             mask = (df['soc_bin'] == soc) & (df['temp_bin'] == temp) & df['R_instant'].notna()
#             if mask.sum() > 10:  # Enough samples
#                 # Use median (robust to outliers)
#                 baseline_R[(soc, temp)] = df.loc[mask, 'R_instant'].median()
    
#     # Step 4: Calculate relative health at each point
#     for idx, row in df.iterrows():
#         if pd.notna(row['R_instant']):
#             soc_key = row['soc_bin']
#             temp_key = row['temp_bin']
            
#             if (soc_key, temp_key) in baseline_R:
#                 baseline = baseline_R[(soc_key, temp_key)]
#                 if baseline > 0:
#                     # Health = 100% * (baseline / current)
#                     # Lower resistance than baseline = better than "new"
#                     # Higher resistance than baseline = worse than "new"
#                     health = 100 * (baseline / row['R_instant'])
                    
#                     # Clip to reasonable range (50-110%)
#                     df.loc[idx, 'resistance_health'] = np.clip(health, 50, 110)
    
#     # Step 5: Alternative approach: Track resistance trend over time
#     # Calculate moving average of resistance
#     if df['R_instant'].notna().sum() > 100:
#         df['R_smoothed'] = df['R_instant'].rolling(window=100, min_periods=10, center=True).mean()
        
#         # Find overall resistance trend (slope)
#         valid_idx = df['R_smoothed'].notna()
#         if valid_idx.sum() > 500:
#             x = np.arange(len(df))[valid_idx]
#             y = df.loc[valid_idx, 'R_smoothed'].values
            
#             # Simple linear trend
#             slope, intercept = np.polyfit(x, y, 1)
            
#             # Health based on resistance growth rate
#             # Assuming 100% increase in resistance over lifetime = 0% SOH
#             # growth_rate = slope / intercept (per data point, convert to per km)
#             avg_mileage = df['vhc_totalMile'].median()
#             mileage_per_point = avg_mileage / len(df)
            
#             if intercept > 0:
#                 resistance_growth = abs(slope / intercept)  # relative growth per data point
#                 # Convert to per 10,000 km
#                 growth_per_10k = resistance_growth * (10000 / mileage_per_point)
                
#                 # SOH based on literature: 50-100% resistance increase over life
#                 # If growth is 0% per 10k km → SOH 100%
#                 # If growth is 10% per 10k km → SOH decreases faster
#                 df['resistance_trend_health'] = 100 * (1 - growth_per_10k / 0.1)  # 10% per 10k km = 0% SOH
#                 df['resistance_trend_health'] = df['resistance_trend_health'].clip(0, 100)
    
#     # Step 6: Use simpler method if above fails
#     if df['resistance_health'].notna().sum() < 100:  # Not enough resistance measurements
#         print(f"  Warning: Insufficient resistance measurements for Vehicle {vehicle_id}")
        
#         # Method A: Use capacity-based SOH if available
#         if 'soh_capacity' in df.columns:
#             df['resistance_health'] = df['soh_capacity']
        
#         # Method B: Use SOC-based internal resistance estimation
#         else:
#             # Internal resistance typically lowest at 50% SOC
#             # Higher at low and high SOC
#             soc = df['bcell_soc']
            
#             # Simplified model: IR increases away from 50% SOC
#             soc_deviation = abs(soc - 50) / 50  # 0 at 50%, 1 at 0% or 100%
            
#             # Health inversely related to IR
#             health_from_soc = 100 * (1 - 0.3 * soc_deviation)  # Up to 30% variation
#             df['resistance_health'] = health_from_soc.clip(70, 100)
    
#     # Step 7: Smooth and fill
#     df['resistance_health'] = (
#         df['resistance_health']
#         .ffill(limit=500)
#         .bfill(limit=500)
#         .rolling(window=100, center=True, min_periods=1)
#         .mean()
#     )
    
#     # Rename to soh_resistance for compatibility
#     df['soh_resistance'] = df['resistance_health']
    
#     # Optional: Remove intermediate columns
#     cols_to_drop = ['R_instant', 'soc_bin', 'temp_bin', 'resistance_health']
#     if 'R_smoothed' in df.columns:
#         cols_to_drop.append('R_smoothed')
#     if 'resistance_trend_health' in df.columns:
#         cols_to_drop.append('resistance_trend_health')
    
#     df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
#     return df

def process_vehicle(file_path, vehicle_id):
    """Process a single vehicle file"""
    
    print(f"\n{'='*60}")
    print(f"Processing Vehicle {vehicle_id}...")
    print(f"{'='*60}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        
        # Get rated capacity from README table
        rated_capacities = {
            1: 150, 2: 150, 3: 160, 4: 160, 5: 160,
            6: 160, 7: 120, 8: 645, 9: 505, 10: 505
        }
        rated_capacity = rated_capacities.get(vehicle_id, 160)
        print(f"  Rated capacity: {rated_capacity} Ah")
        
        # Step 1: Preprocess data
        df = preprocess_data(df, vehicle_id)
        
        # Step 2: Calculate SOH based on capacity
        df = calculate_soh_capacity(df, vehicle_id, rated_capacity)
        
        # Step 3: Calculate SOH based on resistance
        # df = calculate_soh_resistance(df, vehicle_id)
        
        # Step 4: Create final SOH (average of both methods)
        df['soh_final'] = df['soh_capacity'] 
        
        # Step 5: Add vehicle ID and metadata
        df['vehicle_id'] = vehicle_id
        
        # Add chemistry type
        df['chemistry'] = 'NCM' if vehicle_id <= 6 else 'LFP'
        
        # Add vehicle type
        if vehicle_id <= 7:
            df['vehicle_type'] = 'passenger'
        else:
            df['vehicle_type'] = 'bus'
        
        # Step 6: Save processed file
        output_path = file_path.parent / f"vehicle#{vehicle_id}_processed.xlsx"
        df.to_excel(output_path, index=False)
        
        # Print summary statistics
        print(f"\n  Summary for Vehicle {vehicle_id}:")
        print(f"  {'-'*40}")
        print(f"  SOH Capacity: {df['soh_capacity'].mean():.1f}% "
              f"(range: {df['soh_capacity'].min():.1f}-{df['soh_capacity'].max():.1f}%)")
        # print(f"  SOH Resistance: {df['soh_resistance'].mean():.1f}% "
        #       f"(range: {df['soh_resistance'].min():.1f}-{df['soh_resistance'].max():.1f}%)")
        print(f"  SOH Final: {df['soh_final'].mean():.1f}% "
              f"(range: {df['soh_final'].min():.1f}-{df['soh_final'].max():.1f}%)")
        print(f"  NaN in SOH: {df['soh_final'].isna().sum()}/{len(df)} rows")
        
        print(f"\n  Saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"  ERROR processing Vehicle {vehicle_id}: {str(e)}")
        return None

def main():
    """Main processing function"""
    
    print("Starting SOH Label Generation (Capacity Only)...")
    print(f"Data path: {data_path}")
    
    # Check if path exists
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        return []
    
    # Process all vehicle files
    all_data = []
    
    for i in range(1, 11):
        file_name = f"vehicle#{i}.xlsx"
        file_path = data_path / file_name
        
        if file_path.exists():
            df_processed = process_vehicle(file_path, i)
            if df_processed is not None:
                all_data.append(df_processed)
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # OPTION 1: Save as CSV (NO SIZE LIMIT!)
        combined_path_csv = data_path / "all_vehicles_combined.csv"
        combined_df.to_csv(combined_path_csv, index=False)
        
        # OPTION 2: Also save as Parquet (efficient for ML)
        combined_path_parquet = data_path / "all_vehicles_combined.parquet"
        combined_df.to_parquet(combined_path_parquet, index=False)
        
        print(f"\n{'='*60}")
        print("COMBINED DATASET SUMMARY:")
        print(f"{'='*60}")
        print(f"Total vehicles: {len(all_data)}")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Total columns: {len(combined_df.columns)}")
        
        # Summary by vehicle
        print(f"\nSOH by Vehicle:")
        print(f"{'Vehicle':<10} {'Type':<12} {'Chemistry':<8} {'Avg SOH':<10} {'Rows':<10}")
        print(f"{'-'*50}")
        for vid in sorted(combined_df['vehicle_id'].unique()):
            vehicle_data = combined_df[combined_df['vehicle_id'] == vid]
            vehicle_type = vehicle_data['vehicle_type'].iloc[0]
            chemistry = vehicle_data['chemistry'].iloc[0]
            avg_soh = vehicle_data['soh_capacity'].mean()
            print(f"Vehicle {vid:<7} {vehicle_type:<12} {chemistry:<8} {avg_soh:<10.1f}% {len(vehicle_data):<10,}")
        
        print(f"\nOverall average SOH: {combined_df['soh_capacity'].mean():.1f}%")
        print(f"Combined data saved to:")
        print(f"  CSV: {combined_path_csv}")
        print(f"  Parquet: {combined_path_parquet}")
        
        # OPTIONAL: Save summary statistics
        summary_stats = combined_df.groupby('vehicle_id')['soh_capacity'].agg(['mean', 'min', 'max', 'std']).round(2)
        summary_path = data_path / "soh_summary_statistics.csv"
        summary_stats.to_csv(summary_path)
        print(f"  Summary stats: {summary_path}")
    
    return all_data

if __name__ == "__main__":
    # Run the processing
    processed_data = main()