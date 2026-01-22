# file: phase2_soh_label_generation_FIXED_SIMPLE.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set up paths
CLEANED_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(A)")
SOU_LABEL_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(A)\soh_labeled_data")
SOU_LABEL_PATH.mkdir(exist_ok=True, parents=True)

# Initial capacities from README
INITIAL_CAPACITIES = {
    1: 150,  # Ah
    2: 150,
    3: 160,
    4: 160,
    5: 160,
    6: 160,
    8: 645,  # Bus
    9: 505,  # Bus
    10: 505  # Bus
}

# Vehicle types for reference
VEHICLE_TYPES = {
    1: "Passenger",
    2: "Passenger",
    3: "Passenger",
    4: "Passenger",
    5: "Passenger",
    6: "Passenger",
    8: "Bus",
    9: "Bus",
    10: "Bus"
}

def load_cleaned_data(vehicle_num):
    """Load cleaned vehicle data"""
    file_path = CLEANED_PATH / f"vehicle{vehicle_num}_cleaned.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded cleaned Vehicle #{vehicle_num}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Vehicle #{vehicle_num}: {e}")
        return None

def detect_charging_events_simple(df, vehicle_num):
    """
    SIMPLE VERSION: Detect charging events using ONLY charging signal
    This avoids current sign confusion
    """
    print(f"  Detecting charging events for Vehicle #{vehicle_num} (SIMPLE METHOD)...")
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Identify key columns
    time_col = 'time'
    current_col = 'hv_current'
    soc_col = 'bcell_soc'
    speed_col = 'vhc_speed'
    charging_signal_col = 'charging_signal'
    mileage_col = 'vhc_totalmile'
    
    # Verify we have required columns
    required_cols = [time_col, current_col, soc_col, charging_signal_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"    Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Convert to proper numeric types
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[current_col] = pd.to_numeric(df[current_col], errors='coerce')
    df[soc_col] = pd.to_numeric(df[soc_col], errors='coerce')
    df[charging_signal_col] = pd.to_numeric(df[charging_signal_col], errors='coerce')
    
    # SIMPLE METHOD: Use ONLY charging signal
    # Based on README: 1 = charging, 3 = driving
    charging_mask = df[charging_signal_col] == 1
    
    print(f"    Using charging signal only: {charging_mask.sum():,} charging points")
    print(f"    Charging signal value counts: {df[charging_signal_col].value_counts().to_dict()}")
    
    # Group consecutive charging points into events
    charging_groups = (charging_mask != charging_mask.shift()).cumsum()
    
    charging_events = []
    
    for group_id, group_data in df.groupby(charging_groups):
        if charging_mask.iloc[group_data.index[0]]:  # This is a charging group
            if len(group_data) >= 10:  # Minimum 10 samples (100 seconds)
                event = {
                    'vehicle': vehicle_num,
                    'segment_id': group_id,
                    'start_idx': group_data.index[0],
                    'end_idx': group_data.index[-1],
                    'start_time': group_data.iloc[0][time_col],
                    'end_time': group_data.iloc[-1][time_col],
                    'duration_seconds': group_data.iloc[-1][time_col] - group_data.iloc[0][time_col],
                    'duration_hours': (group_data.iloc[-1][time_col] - group_data.iloc[0][time_col]) / 3600,
                    'start_soc': group_data.iloc[0][soc_col],
                    'end_soc': group_data.iloc[-1][soc_col],
                    'delta_soc': group_data.iloc[-1][soc_col] - group_data.iloc[0][soc_col],
                    'avg_current': group_data[current_col].mean(),
                    'max_current': group_data[current_col].max(),
                    'min_current': group_data[current_col].min(),
                    'num_samples': len(group_data),
                    'start_mileage': group_data.iloc[0][mileage_col] if mileage_col in group_data.columns else np.nan,
                    'end_mileage': group_data.iloc[-1][mileage_col] if mileage_col in group_data.columns else np.nan
                }
                
                # SIMPLE VALIDATION - only basic checks
                is_valid = True
                rejection_reason = ""
                
                # Criterion 1: Positive SOC increase (at least 1%)
                if event['delta_soc'] <= 1:
                    is_valid = False
                    rejection_reason = f'No SOC increase ({event["delta_soc"]:.1f}%)'
                
                # Criterion 2: Maximum duration (24 hours) - remove parked vehicles
                elif event['duration_hours'] > 24:
                    is_valid = False
                    rejection_reason = f'Duration too long ({event["duration_hours"]:.1f} hours)'
                
                # Criterion 3: SOC within valid range
                elif event['start_soc'] < 0 or event['start_soc'] > 100 or \
                     event['end_soc'] < 0 or event['end_soc'] > 100:
                    is_valid = False
                    rejection_reason = 'SOC out of range'
                
                # Criterion 4: Minimum duration (2 minutes)
                elif event['duration_hours'] < 0.0333:  # 2 minutes
                    is_valid = False
                    rejection_reason = f'Duration too short ({event["duration_hours"]*60:.1f} min)'
                
                if is_valid:
                    event['rejection_reason'] = 'Accepted'
                    charging_events.append(event)
                else:
                    event['rejection_reason'] = rejection_reason
                    charging_events.append(event)
    
    events_df = pd.DataFrame(charging_events)
    
    if len(events_df) > 0:
        print(f"    Detected {len(events_df)} charging events")
        accepted_count = (events_df['rejection_reason'] == 'Accepted').sum()
        print(f"    Valid events: {accepted_count}")
        
        # Show rejection reasons
        if accepted_count < len(events_df):
            rejection_counts = events_df[events_df['rejection_reason'] != 'Accepted']['rejection_reason'].value_counts()
            for reason, count in rejection_counts.items():
                print(f"      Rejected {reason}: {count}")
    else:
        print(f"    No charging events detected")
    
    return events_df

def detect_charging_events_absolute_current(df, vehicle_num):
    """
    ALTERNATIVE: Detect charging events using ABSOLUTE current (> 1A)
    """
    print(f"  Detecting charging events for Vehicle #{vehicle_num} (ABSOLUTE CURRENT METHOD)...")
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Identify key columns
    time_col = 'time'
    current_col = 'hv_current'
    soc_col = 'bcell_soc'
    speed_col = 'vhc_speed'
    charging_signal_col = 'charging_signal'
    
    # Use charging signal as primary
    charging_mask = pd.Series(False, index=df.index)
    
    if charging_signal_col in df.columns:
        charging_mask = df[charging_signal_col] == 1
        print(f"    Using charging signal: {charging_mask.sum():,} points")
    
    # Also use absolute current when stationary
    if speed_col in df.columns:
        stationary_mask = df[speed_col] < 1.0
        
        # KEY FIX: Use ABSOLUTE current (> 1A) not signed current
        current_mask = np.abs(df[current_col]) > 1.0
        
        combined_mask = stationary_mask & current_mask
        
        charging_mask = charging_mask | combined_mask
        print(f"    Using absolute current (>1A) + stationary: {combined_mask.sum():,} points")
    
    # Group consecutive charging points
    charging_groups = (charging_mask != charging_mask.shift()).cumsum()
    
    charging_events = []
    
    for group_id, group_data in df.groupby(charging_groups):
        if charging_mask.iloc[group_data.index[0]]:
            if len(group_data) >= 10:
                event = {
                    'vehicle': vehicle_num,
                    'segment_id': group_id,
                    'start_idx': group_data.index[0],
                    'end_idx': group_data.index[-1],
                    'start_time': group_data.iloc[0][time_col],
                    'end_time': group_data.iloc[-1][time_col],
                    'duration_seconds': group_data.iloc[-1][time_col] - group_data.iloc[0][time_col],
                    'duration_hours': (group_data.iloc[-1][time_col] - group_data.iloc[0][time_col]) / 3600,
                    'start_soc': group_data.iloc[0][soc_col],
                    'end_soc': group_data.iloc[-1][soc_col],
                    'delta_soc': group_data.iloc[-1][soc_col] - group_data.iloc[0][soc_col],
                    'avg_current': group_data[current_col].mean(),
                    'num_samples': len(group_data),
                }
                
                # VALIDATION with ABSOLUTE current check
                is_valid = True
                
                if event['delta_soc'] <= 1:
                    is_valid = False
                    event['rejection_reason'] = f'No SOC increase ({event["delta_soc"]:.1f}%)'
                elif event['duration_hours'] > 24:
                    is_valid = False
                    event['rejection_reason'] = f'Duration too long ({event["duration_hours"]:.1f}h)'
                elif event['duration_hours'] < 0.0333:
                    is_valid = False
                    event['rejection_reason'] = f'Duration too short ({event["duration_hours"]*60:.1f}min)'
                # KEY FIX: Check ABSOLUTE current magnitude
                elif np.abs(event['avg_current']) < 0.5:
                    is_valid = False
                    event['rejection_reason'] = f'Current too low ({event["avg_current"]:.1f}A)'
                else:
                    event['rejection_reason'] = 'Accepted'
                
                charging_events.append(event)
    
    events_df = pd.DataFrame(charging_events)
    
    if len(events_df) > 0:
        print(f"    Detected {len(events_df)} events, Valid: {(events_df['rejection_reason'] == 'Accepted').sum()}")
    
    return events_df

def calculate_capacity_simple(events_df, df, vehicle_num):
    """
    Calculate battery capacity - SIMPLE version
    """
    print(f"  Calculating capacity for Vehicle #{vehicle_num}...")
    
    if len(events_df) == 0:
        return pd.DataFrame()
    
    # Filter only accepted events
    valid_events = events_df[events_df['rejection_reason'] == 'Accepted'].copy()
    
    if len(valid_events) == 0:
        print(f"    No valid charging events")
        return pd.DataFrame()
    
    capacity_estimates = []
    
    df.columns = [col.strip().lower() for col in df.columns]
    time_col = 'time'
    current_col = 'hv_current'
    
    print(f"    Processing {len(valid_events)} valid charging events...")
    
    for idx, event in valid_events.iterrows():
        segment = df.iloc[event['start_idx']:event['end_idx']+1].copy()
        
        if len(segment) < 2:
            continue
        
        # Ensure numeric
        segment[time_col] = pd.to_numeric(segment[time_col], errors='coerce')
        segment[current_col] = pd.to_numeric(segment[current_col], errors='coerce')
        
        # Calculate time differences in HOURS
        time_seconds = segment[time_col].diff().fillna(0)
        time_hours = time_seconds / 3600.0
        
        # Use ABSOLUTE current (charging can be negative)
        current_abs = np.abs(segment[current_col])
        
        # Simple integration: Ah = Σ(current * Δt)
        integrated_current = 0
        for i in range(len(segment)-1):
            avg_current = (current_abs.iloc[i] + current_abs.iloc[i+1]) / 2
            dt = time_hours.iloc[i+1] if i+1 < len(time_hours) else time_hours.iloc[i]
            integrated_current += avg_current * dt
        
        # Calculate capacity
        delta_soc_fraction = event['delta_soc'] / 100.0
        if delta_soc_fraction > 0.01:  # At least 1% SOC change
            estimated_capacity = integrated_current / delta_soc_fraction
        else:
            estimated_capacity = np.nan
        
        # Calculate SOH
        initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
        if not np.isnan(estimated_capacity) and initial_capacity > 0:
            soh = (estimated_capacity / initial_capacity) * 100.0
        else:
            soh = np.nan
        
        capacity_estimate = {
            'vehicle': vehicle_num,
            'event_id': event['segment_id'],
            'start_time': event['start_time'],
            'end_time': event['end_time'],
            'duration_hours': event['duration_hours'],
            'start_soc': event['start_soc'],
            'end_soc': event['end_soc'],
            'delta_soc': event['delta_soc'],
            'avg_current_a': event['avg_current'],
            'integrated_current_ah': integrated_current,
            'estimated_capacity_ah': estimated_capacity,
            'initial_capacity_ah': initial_capacity,
            'soh_percent': soh,
            'num_samples': event['num_samples']
        }
        
        capacity_estimates.append(capacity_estimate)
    
    estimates_df = pd.DataFrame(capacity_estimates)
    
    if len(estimates_df) > 0:
        print(f"    Calculated {len(estimates_df)} capacity estimates")
        # Show first few
        for i, row in estimates_df.head(3).iterrows():
            print(f"      Event {row['event_id']}: ΔSOC={row['delta_soc']:.1f}%, "
                  f"Capacity={row['estimated_capacity_ah']:.1f}Ah, SOH={row['soh_percent']:.1f}%")
    
    return estimates_df

def filter_estimates_smart(estimates_df, vehicle_num):
    """
    Smart filtering based on data characteristics
    """
    print(f"  Filtering estimates for Vehicle #{vehicle_num}...")
    
    if len(estimates_df) == 0:
        return pd.DataFrame()
    
    initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
    
    # Remove NaN
    filtered = estimates_df.dropna(subset=['estimated_capacity_ah', 'soh_percent']).copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Calculate median for this vehicle
    median_capacity = filtered['estimated_capacity_ah'].median()
    median_soh = filtered['soh_percent'].median()
    
    print(f"    Median capacity: {median_capacity:.1f} Ah, Median SOH: {median_soh:.1f}%")
    
    # Dynamic filtering based on median
    # Allow ±50% around median (wide range for initial filtering)
    capacity_lower = median_capacity * 0.5
    capacity_upper = median_capacity * 1.5
    soh_lower = median_soh * 0.5
    soh_upper = median_soh * 1.5
    
    # Apply filters
    mask = (
        (filtered['estimated_capacity_ah'] >= capacity_lower) &
        (filtered['estimated_capacity_ah'] <= capacity_upper) &
        (filtered['soh_percent'] >= soh_lower) &
        (filtered['soh_percent'] <= soh_upper) &
        (filtered['delta_soc'] >= 5) &
        (filtered['delta_soc'] <= 90) &
        (filtered['duration_hours'] >= 0.0833) &  # 5 min
        (filtered['duration_hours'] <= 12)        # 12 hours
    )
    
    filtered_df = filtered[mask].copy()
    
    print(f"    Kept {len(filtered_df)} of {len(filtered)} estimates after filtering")
    
    return filtered_df

def save_and_plot_results(estimates_df, vehicle_num, df):
    """Save results and create plots"""
    if len(estimates_df) == 0:
        print(f"  No estimates to save for Vehicle #{vehicle_num}")
        return None
    
    # Save estimates
    estimates_file = SOU_LABEL_PATH / f'vehicle{vehicle_num}_capacity_estimates.csv'
    estimates_df.to_csv(estimates_file, index=False)
    print(f"  ✓ Saved capacity estimates: {estimates_file}")
    
    # Create simple plot
    if len(estimates_df) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by time
        estimates_df = estimates_df.sort_values('start_time')
        
        # Plot SOH over time
        if 'start_mileage' in estimates_df.columns and not estimates_df['start_mileage'].isna().all():
            x = estimates_df['start_mileage']
            xlabel = 'Mileage (km)'
        else:
            x = estimates_df['start_time'] / (3600 * 24)  # Days
            xlabel = 'Time (days)'
        
        ax.scatter(x, estimates_df['soh_percent'], alpha=0.6, s=50, color='blue')
        ax.set_title(f'Vehicle #{vehicle_num} - SOH Estimates')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('SOH (%)')
        ax.grid(True, alpha=0.3)
        
        # Add median line
        median_soh = estimates_df['soh_percent'].median()
        ax.axhline(y=median_soh, color='red', linestyle='--', 
                  label=f'Median: {median_soh:.1f}%')
        ax.axhline(y=100, color='green', linestyle=':', label='Initial (100%)')
        ax.legend()
        
        plot_file = SOU_LABEL_PATH / f'vehicle{vehicle_num}_soh_plot.png'
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created SOH plot: {plot_file}")
    
    return estimates_df

def main():
    """Main function - SIMPLE WORKING VERSION"""
    print("=" * 70)
    print("PHASE 2: SOH LABEL GENERATION - SIMPLE WORKING VERSION")
    print("=" * 70)
    print("Using SIMPLE method: charging signal + basic validation")
    print("Avoids current sign confusion")
    print(f"Output will be saved to: {SOU_LABEL_PATH}")
    print()
    
    vehicles_to_process = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    successful_vehicles = 0
    
    for vehicle_num in vehicles_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESSING VEHICLE #{vehicle_num}")
        print(f"{'='*60}")
        
        # Step 1: Load data
        df = load_cleaned_data(vehicle_num)
        if df is None:
            continue
        
        # Step 2: Detect charging events (SIMPLE METHOD)
        charging_events = detect_charging_events_simple(df, vehicle_num)
        
        if len(charging_events) == 0:
            print(f"  ⚠️ No charging events detected")
            
            # Try alternative method
            print(f"  Trying alternative method...")
            charging_events = detect_charging_events_absolute_current(df, vehicle_num)
            
            if len(charging_events) == 0:
                print(f"  ⚠️ Still no charging events")
                continue
        
        # Step 3: Calculate capacity
        capacity_estimates = calculate_capacity_simple(charging_events, df, vehicle_num)
        
        if len(capacity_estimates) == 0:
            print(f"  ⚠️ No capacity estimates calculated")
            continue
        
        # Step 4: Filter estimates
        filtered_estimates = filter_estimates_smart(capacity_estimates, vehicle_num)
        
        if len(filtered_estimates) == 0:
            print(f"  ⚠️ All estimates filtered out")
            # Save raw estimates for debugging
            raw_file = SOU_LABEL_PATH / f'vehicle{vehicle_num}_raw_estimates.csv'
            capacity_estimates.to_csv(raw_file, index=False)
            print(f"  ✓ Saved raw estimates for debugging: {raw_file}")
            continue
        
        # Step 5: Save and plot results
        save_and_plot_results(filtered_estimates, vehicle_num, df)
        
        # Print summary
        print(f"\n  ✅ Vehicle #{vehicle_num} SUCCESS!")
        print(f"    Valid estimates: {len(filtered_estimates)}")
        print(f"    Average SOH: {filtered_estimates['soh_percent'].mean():.1f}%")
        print(f"    Median SOH: {filtered_estimates['soh_percent'].median():.1f}%")
        print(f"    SOH range: {filtered_estimates['soh_percent'].min():.1f}% to {filtered_estimates['soh_percent'].max():.1f}%")
        
        successful_vehicles += 1
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {successful_vehicles} of {len(vehicles_to_process)} vehicles")
    
    if successful_vehicles > 0:
        print(f"\nFiles saved to: {SOU_LABEL_PATH}")
        print("For each successful vehicle:")
        print("  - vehicleX_capacity_estimates.csv (filtered estimates)")
        print("  - vehicleX_soh_plot.png (visualization)")
        print("  - vehicleX_raw_estimates.csv (raw estimates, if filtered out)")
        
        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print("1. Check the SOH values in the CSV files")
        print("2. Verify plots look reasonable")
        print("3. If SOH values are still unrealistic:")
        print("   - Check time units (should be seconds)")
        print("   - Check current integration")
        print("   - Adjust filtering thresholds if needed")
    else:
        print("⚠️ No vehicles were successfully processed")
        print("Check the raw_estimates.csv files to see what values are being calculated")

if __name__ == "__main__":
    main()