
# final_dataset_creation.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_combined_data():
    """Load the cleaned combined dataset."""
    data_path = Path(r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\cleaned_dataset')
    combined_file = data_path / "all_vehicles_combined_cleaned.csv"
    
    if not combined_file.exists():
        print(f"Error: {combined_file} not found!")
        return None
    
    print(f"Loading {combined_file}...")
    df = pd.read_csv(combined_file)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def select_essential_features(df):
    """Select only the essential features for modeling."""
    
    print("\n" + "=" * 60)
    print("SELECTING ESSENTIAL FEATURES")
    print("=" * 60)
    
    # Your selected features
    essential_columns = [
        # Vehicle state
        'vehicle_id',           # Keep for grouping/analysis
        'vhc_speed',
        'vhc_total_mile',
        
        # Battery electrical
        'hv_voltage',
        'hv_current',
        'bcell_soc',
        
        # Cell-level metrics
        'bcell_max_voltage',
        'bcell_min_voltage',
        'bcell_max_temp',
        'bcell_min_temp',
        
        # Operation mode
        'charging_signal',
    ]
    
    # Check which columns exist
    available_essential = [col for col in essential_columns if col in df.columns]
    missing_essential = [col for col in essential_columns if col not in df.columns]
    
    if missing_essential:
        print(f"Warning: Missing essential columns: {missing_essential}")
    
    # Create dataframe with essential features
    essential_df = df[available_essential].copy()
    
    print(f"Selected {len(essential_df.columns)} essential features:")
    for col in essential_df.columns:
        print(f"  ✓ {col}")
    
    return essential_df

def add_derived_features(df):
    """Add calculated derived features."""
    
    print("\n" + "=" * 60)
    print("ADDING DERIVED FEATURES")
    print("=" * 60)
    
    # Create a copy
    enhanced_df = df.copy()
    
    # 1. Power (kW)
    if 'hv_voltage' in enhanced_df.columns and 'hv_current' in enhanced_df.columns:
        enhanced_df['power_kw'] = (enhanced_df['hv_voltage'] * enhanced_df['hv_current']) / 1000
        print(f"  Added: power_kw")
    
    # 2. Cell voltage spread
    if 'bcell_max_voltage' in enhanced_df.columns and 'bcell_min_voltage' in enhanced_df.columns:
        enhanced_df['cell_voltage_spread'] = enhanced_df['bcell_max_voltage'] - enhanced_df['bcell_min_voltage']
        print(f"  Added: cell_voltage_spread")
    
    # 3. Cell temperature spread
    if 'bcell_max_temp' in enhanced_df.columns and 'bcell_min_temp' in enhanced_df.columns:
        enhanced_df['cell_temp_spread'] = enhanced_df['bcell_max_temp'] - enhanced_df['bcell_min_temp']
        print(f"  Added: cell_temp_spread")
    
    # 4. Operation flags (optional but useful)
    if 'charging_signal' in enhanced_df.columns:
        enhanced_df['is_charging'] = (enhanced_df['charging_signal'] == 1).astype(int)
        enhanced_df['is_driving'] = (enhanced_df['charging_signal'] == 3).astype(int)
        print(f"  Added: is_charging, is_driving")
    
    return enhanced_df

def create_soh_labels(df):
    """
    Create ACTUAL SOH labels using physics-based calculations.
    
    1. Capacity-based SOH: Using ∆Q/∆SOC method from charge/discharge cycles
    2. Resistance-based SOH: Using voltage sag during acceleration events
    """
    
    print("\n" + "=" * 60)
    print("CALCULATING REAL SOH LABELS")
    print("=" * 60)
    
    labeled_df = df.copy()
    
    # We need metadata for initial capacities
    VEHICLE_METADATA = {
        1: {"capacity_ah": 150, "type": "passenger", "chemistry": "NCM"},
        2: {"capacity_ah": 150, "type": "passenger", "chemistry": "NCM"},
        3: {"capacity_ah": 160, "type": "passenger", "chemistry": "NCM"},
        4: {"capacity_ah": 160, "type": "passenger", "chemistry": "NCM"},
        5: {"capacity_ah": 160, "type": "passenger", "chemistry": "NCM"},
        6: {"capacity_ah": 160, "type": "passenger", "chemistry": "NCM"},
        7: {"capacity_ah": 120, "type": "passenger", "chemistry": "LFP"},
        8: {"capacity_ah": 645, "type": "bus", "chemistry": "LFP"},
        9: {"capacity_ah": 505, "type": "bus", "chemistry": "LFP"},
        10: {"capacity_ah": 505, "type": "bus", "chemistry": "LFP"},
    }
    
    # Initialize SOH columns
    labeled_df['soh_capacity'] = np.nan
    labeled_df['soh_resistance'] = np.nan
    
    if 'vehicle_id' not in labeled_df.columns or 'time' not in labeled_df.columns:
        print("Error: Need vehicle_id and time columns for SOH calculation")
        return labeled_df
    
    print("Calculating SOH for each vehicle...")
    
    for vehicle_id in sorted(labeled_df['vehicle_id'].unique()):
        print(f"\n--- Vehicle {vehicle_id} ---")
        
        # Get vehicle data sorted by time
        vehicle_mask = labeled_df['vehicle_id'] == vehicle_id
        vehicle_data = labeled_df[vehicle_mask].copy()
        vehicle_data = vehicle_data.sort_values('time')
        
        if len(vehicle_data) < 100:  # Need enough data
            print(f"  Skipping: insufficient data ({len(vehicle_data)} rows)")
            continue
        
        # Get initial capacity from metadata
        initial_capacity = VEHICLE_METADATA.get(vehicle_id, {}).get('capacity_ah', 150)
        
        # ============================================================
        # 1. CAPACITY-BASED SOH CALCULATION (∆Q/∆SOC method)
        # ============================================================
        
        print("  Calculating capacity-based SOH...")
        
        # We need to find complete charge or discharge cycles
        capacity_estimates = []
        
        # Method A: Look for charging cycles (SOC increasing)
        if 'bcell_soc' in vehicle_data.columns and 'hv_current' in vehicle_data.columns:
            
            # Find periods of significant SOC change (at least 20% change)
            soc_diff = vehicle_data['bcell_soc'].diff().abs()
            
            # Look for continuous periods with significant SOC change
            charge_events = []
            discharge_events = []
            
            current_event = []
            min_soc_in_event = None
            max_soc_in_event = None
            
            for idx, row in vehicle_data.iterrows():
                # Start new event if:
                # 1. First row
                # 2. Large time gap (> 3600 seconds = 1 hour)
                # 3. Current changes sign (switching between charge/discharge)
                
                if len(current_event) == 0:
                    current_event.append(idx)
                    min_soc_in_event = row['bcell_soc']
                    max_soc_in_event = row['bcell_soc']
                else:
                    time_gap = row['time'] - vehicle_data.loc[current_event[-1], 'time']
                    prev_current = vehicle_data.loc[current_event[-1], 'hv_current']
                    
                    # Check if we should start a new event
                    if (time_gap > 3600 or 
                        (prev_current > 0 and row['hv_current'] < -10) or  # Switching to discharge
                        (prev_current < 0 and row['hv_current'] > 10)):    # Switching to charge
                        
                        # Save completed event if significant
                        soc_range = max_soc_in_event - min_soc_in_event if min_soc_in_event and max_soc_in_event else 0
                        if soc_range >= 20:  # At least 20% SOC change
                            # Determine if charge or discharge
                            avg_current = vehicle_data.loc[current_event, 'hv_current'].mean()
                            if avg_current < 0:  # Negative current = charging
                                charge_events.append(current_event.copy())
                            else:  # Positive current = discharging
                                discharge_events.append(current_event.copy())
                        
                        # Start new event
                        current_event = [idx]
                        min_soc_in_event = row['bcell_soc']
                        max_soc_in_event = row['bcell_soc']
                    else:
                        current_event.append(idx)
                        min_soc_in_event = min(min_soc_in_event, row['bcell_soc'])
                        max_soc_in_event = max(max_soc_in_event, row['bcell_soc'])
            
            # Process charging events (more reliable for capacity estimation)
            for event_indices in charge_events:
                event_data = vehicle_data.loc[event_indices]
                
                # Skip if too short
                if len(event_data) < 10:
                    continue
                
                # Calculate total charge transferred (Coulomb counting)
                # Q = ∫ I dt (convert seconds to hours: ÷ 3600)
                # Current is negative during charging, so take absolute value
                time_diff = event_data['time'].diff().fillna(0)
                charge_transferred = (abs(event_data['hv_current']) * time_diff / 3600).sum()  # Ah
                
                # Calculate SOC change
                soc_start = event_data['bcell_soc'].iloc[0]
                soc_end = event_data['bcell_soc'].iloc[-1]
                soc_change = soc_end - soc_start
                
                # Calculate capacity: Capacity = ∆Q / ∆SOC
                if soc_change > 5:  # Minimum 5% SOC change for reliable estimate
                    capacity_est = charge_transferred / (soc_change / 100)  # Convert SOC% to fraction
                    capacity_estimates.append(capacity_est)
                    
                    # Calculate SOH for this event
                    soh_capacity_event = capacity_est / initial_capacity
                    labeled_df.loc[event_indices, 'soh_capacity'] = soh_capacity_event
            
            print(f"    Found {len(charge_events)} charging events, {len(capacity_estimates)} valid capacity estimates")
            
            # If we have capacity estimates, fill forward for the whole vehicle
            if len(capacity_estimates) > 0:
                # Use median capacity estimate for this vehicle
                median_capacity = np.median(capacity_estimates)
                median_soh = median_capacity / initial_capacity
                
                # Fill all vehicle rows with the median estimate
                labeled_df.loc[vehicle_mask, 'soh_capacity'] = median_soh
                
                print(f"    Estimated capacity: {median_capacity:.1f} Ah")
                print(f"    Capacity SOH: {median_soh:.3f} ({median_soh*100:.1f}%)")
        
        # ============================================================
        # 2. RESISTANCE-BASED SOH CALCULATION (∆V/∆I method)
        # ============================================================
        
        print("  Calculating resistance-based SOH...")
        
        resistance_estimates = []
        
        if ('hv_voltage' in vehicle_data.columns and 
            'hv_current' in vehicle_data.columns and 
            'vhc_speed' in vehicle_data.columns):
            
            # Look for acceleration/deceleration events
            # These create sudden current changes ideal for resistance calculation
            
            # Calculate current changes (absolute)
            vehicle_data['current_diff'] = vehicle_data['hv_current'].diff().abs()
            vehicle_data['voltage_diff'] = vehicle_data['hv_voltage'].diff()
            
            # Find significant current steps (> 20A change within 10 seconds)
            # This indicates acceleration or regenerative braking
            time_diff = vehicle_data['time'].diff()
            current_rate_of_change = vehicle_data['current_diff'] / time_diff.clip(lower=1)
            
            # Find events with rapid current change
            rapid_change_mask = (current_rate_of_change.abs() > 2)  # > 2 A/s change
            
            # For each rapid change, calculate instantaneous resistance
            for idx in vehicle_data[rapid_change_mask].index:
                # Get window around the event
                start_idx = max(vehicle_data.index.get_loc(idx) - 5, 0)
                end_idx = min(vehicle_data.index.get_loc(idx) + 5, len(vehicle_data) - 1)
                
                window_indices = vehicle_data.index[start_idx:end_idx+1]
                window_data = vehicle_data.loc[window_indices]
                
                # Find minimum voltage and maximum current in the window
                # (for discharge events - acceleration)
                if window_data['hv_current'].max() > 20:  # Significant discharge current
                    min_voltage_idx = window_data['hv_voltage'].idxmin()
                    max_current_idx = window_data['hv_current'].idxmax()
                    
                    # Check if they're close in time (< 5 seconds)
                    time_gap = abs(window_data.loc[min_voltage_idx, 'time'] - 
                                  window_data.loc[max_current_idx, 'time'])
                    
                    if time_gap < 5:
                        DV = vehicle_data.loc[min_voltage_idx, 'hv_voltage'] - vehicle_data.loc[max_current_idx, 'hv_voltage']
                        DI = vehicle_data.loc[max_current_idx, 'hv_current'] - vehicle_data.loc[min_voltage_idx, 'hv_current']
                        
                        if DI > 10:  # Significant current change
                            resistance = abs(DV / DI)  # Ohms
                            resistance_estimates.append(resistance)
            
            print(f"    Found {len(resistance_estimates)} resistance measurement events")
            
            # Calculate resistance-based SOH
            if len(resistance_estimates) > 0:
                # Initial resistance estimates (typical values)
                # NCM: ~1-2 mΩ per cell, LFP: ~2-3 mΩ per cell
                chemistry = VEHICLE_METADATA.get(vehicle_id, {}).get('chemistry', 'NCM')
                
                if chemistry == 'NCM':
                    initial_resistance_per_cell = 1.5  # mΩ
                else:  # LFP
                    initial_resistance_per_cell = 2.5  # mΩ
                
                # Get number of cells in series
                series_cells = VEHICLE_METADATA.get(vehicle_id, {}).get('series_cells', 91)
                initial_pack_resistance = initial_resistance_per_cell * series_cells / 1000  # Convert to Ω
                
                # Calculate median current resistance
                median_resistance = np.median(resistance_estimates)
                
                # Resistance SOH = Initial / Current (resistance increases as battery ages)
                if median_resistance > 0:
                    soh_resistance = initial_pack_resistance / median_resistance
                    soh_resistance = np.clip(soh_resistance, 0.5, 1.2)  # Reasonable bounds
                    
                    # Assign to vehicle
                    labeled_df.loc[vehicle_mask, 'soh_resistance'] = soh_resistance
                    
                    print(f"    Estimated pack resistance: {median_resistance*1000:.1f} mΩ")
                    print(f"    Resistance SOH: {soh_resistance:.3f}")
        
        # Fill missing SOH values with forward fill for each vehicle
        vehicle_mask = labeled_df['vehicle_id'] == vehicle_id
        
        # Forward fill capacity SOH within each vehicle
        if labeled_df.loc[vehicle_mask, 'soh_capacity'].notna().any():
            labeled_df.loc[vehicle_mask, 'soh_capacity'] = (
                labeled_df.loc[vehicle_mask, 'soh_capacity'].ffill()
            )
        
        # Forward fill resistance SOH within each vehicle
        if labeled_df.loc[vehicle_mask, 'soh_resistance'].notna().any():
            labeled_df.loc[vehicle_mask, 'soh_resistance'] = (
                labeled_df.loc[vehicle_mask, 'soh_resistance'].ffill()
            )
    
    # ============================================================
    # 3. FINAL PROCESSING AND VALIDATION
    # ============================================================
    
    print("\n" + "=" * 60)
    print("SOH CALCULATION SUMMARY")
    print("=" * 60)
    
    # Calculate statistics
    capacity_coverage = labeled_df['soh_capacity'].notna().sum() / len(labeled_df) * 100
    resistance_coverage = labeled_df['soh_resistance'].notna().sum() / len(labeled_df) * 100
    
    print(f"\nSOH Coverage:")
    print(f"  Capacity SOH: {capacity_coverage:.1f}% of rows")
    print(f"  Resistance SOH: {resistance_coverage:.1f}% of rows")
    
    # Only show statistics if we have enough data
    if capacity_coverage > 10:
        valid_capacity = labeled_df['soh_capacity'].dropna()
        print(f"\nCapacity SOH Statistics:")
        print(f"  Min: {valid_capacity.min():.3f}")
        print(f"  Max: {valid_capacity.max():.3f}")
        print(f"  Mean: {valid_capacity.mean():.3f}")
        print(f"  Std: {valid_capacity.std():.3f}")
    
    if resistance_coverage > 10:
        valid_resistance = labeled_df['soh_resistance'].dropna()
        print(f"\nResistance SOH Statistics:")
        print(f"  Min: {valid_resistance.min():.3f}")
        print(f"  Max: {valid_resistance.max():.3f}")
        print(f"  Mean: {valid_resistance.mean():.3f}")
        print(f"  Std: {valid_resistance.std():.3f}")
    
    # Check correlation between capacity and resistance SOH
    both_valid = labeled_df[['soh_capacity', 'soh_resistance']].dropna()
    if len(both_valid) > 10:
        correlation = both_valid['soh_capacity'].corr(both_valid['soh_resistance'])
        print(f"\nCorrelation between Capacity and Resistance SOH: {correlation:.3f}")
        
        if correlation > 0.5:
            print("  ✓ Strong positive correlation (expected: both degrade together)")
        elif correlation > 0.2:
            print("  ○ Moderate correlation")
        else:
            print("  ⚠️ Weak correlation - check calculations")
    
    # Fill any remaining NaN with reasonable defaults
    # But only if we have at least some valid data
    overall_capacity_mean = labeled_df['soh_capacity'].mean()
    overall_resistance_mean = labeled_df['soh_resistance'].mean()
    
    if not np.isnan(overall_capacity_mean):
        labeled_df['soh_capacity'] = labeled_df['soh_capacity'].fillna(overall_capacity_mean)
    
    if not np.isnan(overall_resistance_mean):
        labeled_df['soh_resistance'] = labeled_df['soh_resistance'].fillna(overall_resistance_mean)
    
    return labeled_df

def clean_final_dataset(df):
    """Final cleaning of the dataset."""
    
    print("\n" + "=" * 60)
    print("FINAL CLEANING")
    print("=" * 60)
    
    # 1. Remove any rows with NaN in critical columns
    critical_cols = ['hv_voltage', 'hv_current', 'bcell_soc', 'vhc_speed']
    critical_cols = [col for col in critical_cols if col in df.columns]
    
    initial_rows = len(df)
    df_clean = df.dropna(subset=critical_cols)
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        print(f"Removed {removed} rows with missing critical data ({removed/initial_rows*100:.1f}%)")
    
    # 2. Remove infinite values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # 3. Remove any remaining NaN rows in SOH labels if they exist
    if 'soh_capacity' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['soh_capacity', 'soh_resistance'])
    
    # 4. Ensure reasonable value ranges
    if 'bcell_soc' in df_clean.columns:
        # SOC should be 0-100%
        df_clean = df_clean[(df_clean['bcell_soc'] >= 0) & (df_clean['bcell_soc'] <= 100)]
    
    if 'vhc_speed' in df_clean.columns:
        # Reasonable speed limits
        df_clean = df_clean[df_clean['vhc_speed'] >= 0]
        df_clean = df_clean[df_clean['vhc_speed'] <= 200]  # 200 km/h max
    
    print(f"Final dataset: {len(df_clean):,} rows, {len(df_clean.columns):,} columns")
    
    return df_clean

def save_final_dataset(df, output_dir):
    """Save the final dataset."""
    
    print("\n" + "=" * 60)
    print("SAVING FINAL DATASET")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir) / "combined_dataset"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    csv_path = output_path / "final_ev_dataset.csv"
    parquet_path = output_path / "final_ev_dataset.parquet"
    excel_path = output_path / "final_ev_dataset.xlsx"
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save Parquet (efficient for ML)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")
    
    # Save Excel (for manual inspection)
    # Limit to first 100k rows for Excel
    if len(df) > 100000:
        df.head(100000).to_excel(excel_path, index=False)
        print(f"Saved Excel (first 100k rows): {excel_path}")
    else:
        df.to_excel(excel_path, index=False)
        print(f"Saved Excel: {excel_path}")
    
    # Create a data dictionary
    data_dict = {
        'vehicle_id': 'Vehicle identifier (1-10)',
        'vhc_speed': 'Vehicle speed (km/h)',
        'vhc_total_mile': 'Cumulative mileage (km) - aging indicator',
        'hv_voltage': 'Battery pack total voltage (V)',
        'hv_current': 'Battery pack current (A), negative=charging',
        'bcell_soc': 'State of Charge (%)',
        'bcell_max_voltage': 'Maximum cell voltage (V)',
        'bcell_min_voltage': 'Minimum cell voltage (V)',
        'bcell_max_temp': 'Maximum cell temperature (°C)',
        'bcell_min_temp': 'Minimum cell temperature (°C)',
        'charging_signal': '1=charging, 3=driving/not charging',
        'power_kw': 'Instantaneous power (kW) = voltage × current / 1000',
        'cell_voltage_spread': 'Voltage imbalance = max_voltage - min_voltage',
        'cell_temp_spread': 'Temperature gradient = max_temp - min_temp',
        'is_charging': 'Boolean (1=charging, 0=not charging)',
        'is_driving': 'Boolean (1=driving, 0=not driving)',
        'soh_capacity': 'Estimated State of Health - Capacity based (0-1)',
        'soh_resistance': 'Estimated State of Health - Resistance based (0-1)'
    }
    
    # Save data dictionary
    dict_df = pd.DataFrame(list(data_dict.items()), columns=['column', 'description'])
    dict_path = output_path / "data_dictionary.csv"
    dict_df.to_csv(dict_path, index=False)
    print(f"Saved data dictionary: {dict_path}")
    
    return output_path

def analyze_final_dataset(df):
    """Provide analysis of the final dataset."""
    
    print("\n" + "=" * 60)
    print("FINAL DATASET ANALYSIS")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    print("\n1. Column Summary:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"  {col}: {dtype}, {non_null:,} non-null ({pct:.1f}%)")
    
    print("\n2. Vehicle Distribution:")
    if 'vehicle_id' in df.columns:
        vehicle_dist = df['vehicle_id'].value_counts().sort_index()
        for vid, count in vehicle_dist.items():
            pct = count / len(df) * 100
            print(f"  Vehicle {vid}: {count:,} rows ({pct:.1f}%)")
    
    print("\n3. SOH Label Statistics:")
    if 'soh_capacity' in df.columns:
        print(f"  Capacity SOH: min={df['soh_capacity'].min():.3f}, "
              f"max={df['soh_capacity'].max():.3f}, "
              f"mean={df['soh_capacity'].mean():.3f}")
        
        print(f"  Resistance SOH: min={df['soh_resistance'].min():.3f}, "
              f"max={df['soh_resistance'].max():.3f}, "
              f"mean={df['soh_resistance'].mean():.3f}")
    
    print("\n4. Operation Modes:")
    if 'charging_signal' in df.columns:
        modes = df['charging_signal'].value_counts()
        for mode, count in modes.items():
            pct = count / len(df) * 100
            mode_name = {1: 'Charging', 3: 'Driving'}.get(mode, f'Unknown ({mode})')
            print(f"  {mode_name}: {count:,} ({pct:.1f}%)")
    
    print("\n5. Correlation with SOH (top 5):")
    if 'soh_capacity' in df.columns:
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['soh_capacity'].abs().sort_values(ascending=False)
        print("  Most correlated features with Capacity SOH:")
        for idx, (feature, corr) in enumerate(correlations.items()[:6]):
            if feature != 'soh_capacity' and feature != 'soh_resistance':
                print(f"    {feature}: {corr:.3f}")

def main():
    """Main pipeline to create final dataset."""
    
    print("=" * 60)
    print("FINAL DATASET CREATION PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    df = load_combined_data()
    if df is None:
        return
    
    # 2. Select essential features
    essential_df = select_essential_features(df)
    
    # 3. Add derived features
    enhanced_df = add_derived_features(essential_df)
    
    # 4. Create SOH labels
    labeled_df = create_soh_labels(enhanced_df)
    
    # 5. Clean final dataset
    final_df = clean_final_dataset(labeled_df)
    
    # 6. Save final dataset
    output_dir = r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)'
    saved_path = save_final_dataset(final_df, output_dir)
    
    # 7. Analyze final dataset
    analyze_final_dataset(final_df)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE! 🎉")
    print("=" * 60)
    
    print(f"\nFinal dataset saved in: {saved_path}")
    print("\nFiles created:")
    print("  ✓ final_ev_dataset.csv      - Full dataset in CSV format")
    print("  ✓ final_ev_dataset.parquet  - Efficient binary format for ML")
    print("  ✓ final_ev_dataset.xlsx     - Excel file for manual inspection")
    print("  ✓ data_dictionary.csv       - Documentation of all columns")
    
    print("\nDataset ready for:")
    print("  • Machine learning model training")
    print("  • Battery health prediction")
    print("  • Feature importance analysis")
    print("  • Cross-vehicle comparisons")

if __name__ == "__main__":
    main()