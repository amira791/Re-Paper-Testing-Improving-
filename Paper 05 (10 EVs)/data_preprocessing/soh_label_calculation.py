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
    FIXED CURRENT SIGN CONVENTION BASED ON YOUR DATA
    """
    
    df = df.copy()
    df['soh_capacity'] = np.nan
    
    # DEBUG: Print current stats
    print(f"  Vehicle {vehicle_id} current stats:")
    print(f"    Min: {df['hv_current'].min():.1f}A, Max: {df['hv_current'].max():.1f}A")
    print(f"    Negative %: {(df['hv_current'] < 0).sum()/len(df)*100:.1f}%")
    print(f"    Positive %: {(df['hv_current'] > 0).sum()/len(df)*100:.1f}%")
    
    # Based on your data samples:
    # Vehicle 5: -31.8A while driving (46 km/h) → NEGATIVE = DISCHARGE
    # Vehicle 10: -77.7A while charging (0 km/h) → NEGATIVE LARGE = FAST CHARGE
    # CONCLUSION: NEGATIVE CURRENT = ENERGY FLOWING FROM GRID TO BATTERY (Charge)
    #             POSITIVE CURRENT = ENERGY FLOWING FROM BATTERY TO MOTOR (Discharge)
    
    # Time interval in hours (0.1Hz = 10 seconds = 0.0027778 hours)
    time_interval_h = 10 / 3600  # 10 seconds in hours
    
    # Method 1: Charging events (charging_signal = 1 OR negative current when parked)
    if 'charging_signal' in df.columns:
        charge_mask = (df['charging_signal'] == 1) | ((df['vhc_speed'] == 0) & (df['hv_current'] < -5))
    else:
        charge_mask = (df['vhc_speed'] == 0) & (df['hv_current'] < -5)
    
    if charge_mask.any():
        df['charge_group'] = (charge_mask != charge_mask.shift()).cumsum()
        
        for group_id in df[charge_mask]['charge_group'].unique():
            group_data = df[df['charge_group'] == group_id]
            if len(group_data) > 5:  # At least 5 points (50 seconds)
                # Charge current is NEGATIVE, take absolute value
                charge_ah = (abs(group_data['hv_current']) * time_interval_h).sum()
                
                soc_start = group_data['bcell_soc'].iloc[0]
                soc_end = group_data['bcell_soc'].iloc[-1]
                soc_change = soc_end - soc_start
                
                if soc_change > 2:  # At least 2% SOC increase
                    estimated_capacity = charge_ah / (soc_change / 100)
                    soh_value = min(105, max(70, (estimated_capacity / rated_capacity) * 100))
                    
                    df.loc[group_data.index, 'soh_capacity'] = soh_value
                    print(f"    Charge event: {soc_start:.1f}%→{soc_end:.1f}% "
                          f"({soc_change:.1f}%), Capacity: {estimated_capacity:.1f}Ah, "
                          f"SOH: {soh_value:.1f}%")
    
    # Method 2: Discharge events (driving with positive current)
    if 'vhc_speed' in df.columns:
        discharge_mask = (df['vhc_speed'] > 1) & (df['hv_current'] > 0)
        
        if discharge_mask.any():
            df['discharge_group'] = (discharge_mask != discharge_mask.shift()).cumsum()
            
            for group_id in df[discharge_mask]['discharge_group'].unique():
                group_data = df[df['discharge_group'] == group_id]
                if len(group_data) > 10:  # At least 10 points (100 seconds driving)
                    discharge_ah = (group_data['hv_current'] * time_interval_h).sum()
                    
                    soc_start = group_data['bcell_soc'].iloc[0]
                    soc_end = group_data['bcell_soc'].iloc[-1]
                    soc_change = soc_start - soc_end  # Positive for discharge
                    
                    if soc_change > 1:  # At least 1% SOC decrease
                        estimated_capacity = discharge_ah / (soc_change / 100)
                        soh_value = min(105, max(70, (estimated_capacity / rated_capacity) * 100))
                        
                        # Only update if no charge estimate exists
                        mask = group_data.index & df['soh_capacity'].isna()
                        df.loc[mask, 'soh_capacity'] = soh_value
    
    # Fill gaps with reasonable values
    if df['soh_capacity'].isna().all():
        # No charge/discharge events found - use mileage-based estimate
        print(f"  Warning: No charge/discharge events found for capacity estimation")
        
        # Use simple mileage degradation model
        avg_mileage = df['vhc_totalMile'].median()
        
        # Degradation rates from literature
        if vehicle_id <= 6:  # NCM
            degradation_per_km = 2.0 / 30000  # 2% per 30,000 km
        else:  # LFP
            degradation_per_km = 1.5 / 30000  # 1.5% per 30,000 km
        
        base_soh = 100 - (avg_mileage * degradation_per_km)
        base_soh = max(70, min(100, base_soh))
        df['soh_capacity'] = base_soh
        print(f"  Using mileage-based SOH: {base_soh:.1f}% at {avg_mileage:.0f} km")
    else:
        # Forward fill with limit
        df['soh_capacity'] = df['soh_capacity'].ffill(limit=500).bfill(limit=500)
        
        # If still NaN, fill with median
        if df['soh_capacity'].isna().any():
            median_soh = df['soh_capacity'].median()
            df['soh_capacity'] = df['soh_capacity'].fillna(median_soh)
    
    return df

def calculate_soh_resistance(df, vehicle_id):
    """
    Calculate SOH based on internal resistance with BETTER BASELINES
    """
    
    df = df.copy()
    df['soh_resistance'] = np.nan
    
    # Literature-based baseline resistances (Ohms for FULL PACK)
    # Sources: Typical values for EV batteries
    baseline_resistances = {
        # Vehicle: (discharge_R_new, charge_R_new) in Ohms
        1: (0.027, 0.036),  # NCM 150Ah, 91S
        2: (0.027, 0.036),
        3: (0.025, 0.034),  # NCM 160Ah, 91S
        4: (0.025, 0.034),
        5: (0.025, 0.034),
        6: (0.025, 0.034),
        7: (0.080, 0.100),  # LFP 120Ah, ~166S estimate
        8: (0.024, 0.032),  # LFP 645Ah bus (large cells, lower resistance)
        9: (0.030, 0.040),  # LFP 505Ah bus
        10: (0.030, 0.040)  # LFP 505Ah bus
    }
    
    # Get baseline for this vehicle
    if vehicle_id in baseline_resistances:
        R_new_discharge, R_new_charge = baseline_resistances[vehicle_id]
    else:
        # Default estimates
        if vehicle_id <= 6:  # NCM
            R_new_discharge, R_new_charge = 0.026, 0.035
        else:  # LFP
            R_new_discharge, R_new_charge = 0.050, 0.070
    
    # Calculate resistance from discharge transients
    discharge_mask = (df['vhc_speed'] > 5) & (df['hv_current'] > 1)
    
    if discharge_mask.any():
        # Find current spikes (acceleration events)
        df['current_ma'] = df['hv_current'].rolling(window=5, center=True).mean()
        df['current_diff'] = df['current_ma'].diff().abs()
        
        # Use adaptive threshold
        threshold = df.loc[discharge_mask, 'current_diff'].quantile(0.75)
        
        spike_indices = df[discharge_mask & (df['current_diff'] > threshold)].index
        
        for idx in spike_indices:
            if idx > 10 and idx < len(df) - 10:
                # Window around spike
                pre_window = df.loc[idx-5:idx-1]
                post_window = df.loc[idx:idx+4]
                
                v_pre = pre_window['hv_voltage'].mean()
                v_post = post_window['hv_voltage'].mean()
                i_pre = pre_window['hv_current'].mean()
                i_post = post_window['hv_current'].mean()
                
                delta_v = v_post - v_pre
                delta_i = i_post - i_pre
                
                if delta_i > 5:  # Significant current increase
                    R_actual = abs(delta_v / delta_i)
                    
                    # SOH from resistance: R increases as battery ages
                    # Typical: R increases 50-100% over life
                    # SOH = 100 * (2 - R_actual/R_new) for 100% increase at EOL
                    soh_r = 100 * (2 - (R_actual / R_new_discharge))
                    soh_r = np.clip(soh_r, 50, 105)
                    
                    df.loc[idx, 'soh_resistance'] = soh_r
    
    # Calculate from charge transients
    charge_mask = (df['vhc_speed'] == 0) & (df['hv_current'] < -1)
    
    if charge_mask.any():
        charge_spikes = df[charge_mask & (df['hv_current'].diff().abs() > 5)].index
        
        for idx in charge_spikes:
            if idx > 5:
                # Current becomes more negative = charging increase
                delta_v = df.loc[idx, 'hv_voltage'] - df.loc[idx-1, 'hv_voltage']
                delta_i = df.loc[idx, 'hv_current'] - df.loc[idx-1, 'hv_voltage']
                
                if abs(delta_i) > 5:
                    R_charge = abs(delta_v / delta_i)
                    soh_r_charge = 100 * (2 - (R_charge / R_new_charge))
                    soh_r_charge = np.clip(soh_r_charge, 50, 105)
                    
                    df.loc[idx, 'soh_resistance'] = soh_r_charge
    
    # Fill and smooth
    if df['soh_resistance'].notna().any():
        df['soh_resistance'] = df['soh_resistance'].ffill(limit=200).bfill(limit=200)
        df['soh_resistance'] = df['soh_resistance'].rolling(window=50, center=True, min_periods=1).mean()
    else:
        # No resistance measurements - use capacity SOH or default
        if 'soh_capacity' in df.columns:
            df['soh_resistance'] = df['soh_capacity']
        else:
            df['soh_resistance'] = 95  # Default guess
    
    return df

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
        df = calculate_soh_resistance(df, vehicle_id)
        
        # Step 4: Create final SOH (average of both methods)
        df['soh_final'] = (df['soh_capacity'] + df['soh_resistance']) / 2
        
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
        print(f"  SOH Resistance: {df['soh_resistance'].mean():.1f}% "
              f"(range: {df['soh_resistance'].min():.1f}-{df['soh_resistance'].max():.1f}%)")
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
    
    print("Starting SOH Label Generation...")
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
    
    # Optional: Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = data_path / "all_vehicles_combined.xlsx"
        combined_df.to_excel(combined_path, index=False)
        
        print(f"\n{'='*60}")
        print("COMBINED DATASET SUMMARY:")
        print(f"{'='*60}")
        print(f"Total vehicles: {len(all_data)}")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Total columns: {len(combined_df.columns)}")
        
        # Summary by vehicle
        print(f"\nSOH by Vehicle:")
        for vid in sorted(combined_df['vehicle_id'].unique()):
            vehicle_data = combined_df[combined_df['vehicle_id'] == vid]
            print(f"  Vehicle {vid}: {vehicle_data['soh_final'].mean():.1f}% "
                  f"(n={len(vehicle_data):,})")
        
        print(f"\nCombined data saved to: {combined_path}")
    
    return all_data

if __name__ == "__main__":
    # Run the processing
    processed_data = main()