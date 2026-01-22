# file: phase2_soh_label_generation_FIXED.py
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

def analyze_time_interval(df, vehicle_num):
    """Analyze time intervals to understand time units"""
    print(f"  Analyzing time intervals for Vehicle #{vehicle_num}...")
    
    # Find time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    
    if not time_col:
        print("    No time column found")
        return None
    
    # Calculate time differences
    time_diffs = df[time_col].diff().dropna()
    
    # Get statistics
    median_diff = time_diffs.median()
    mean_diff = time_diffs.mean()
    min_diff = time_diffs.min()
    max_diff = time_diffs.max()
    
    print(f"    Time interval statistics:")
    print(f"      Median: {median_diff:.3f} units")
    print(f"      Mean: {mean_diff:.3f} units")
    print(f"      Range: {min_diff:.3f} to {max_diff:.3f} units")
    
    # Determine likely time unit
    if 9.9 <= median_diff <= 10.1:
        print(f"    ✓ Likely seconds (0.1 Hz sampling)")
        return 'seconds'
    elif 0.09 <= median_diff <= 0.11:
        print(f"    ⚠️ Likely 0.1 seconds (10 Hz sampling)")
        return 'tenths_of_seconds'
    elif 599 <= median_diff <= 601:
        print(f"    ⚠️ Likely minutes (0.00167 Hz sampling)")
        return 'minutes'
    else:
        print(f"    ⚠️ Unknown time unit: {median_diff:.3f}")
        return 'unknown'

def analyze_current_sign(df, vehicle_num):
    """Analyze current sign convention"""
    print(f"  Analyzing current sign for Vehicle #{vehicle_num}...")
    
    # Find current column
    current_col = None
    for col in df.columns:
        if 'current' in col.lower() and 'bcell' not in col.lower():
            current_col = col
            break
    
    if not current_col:
        print("    No current column found")
        return None
    
    # Get statistics
    current_stats = {
        'min': df[current_col].min(),
        'max': df[current_col].max(),
        'mean': df[current_col].mean(),
        'std': df[current_col].std(),
        'negative_pct': (df[current_col] < 0).sum() / len(df) * 100,
        'positive_pct': (df[current_col] > 0).sum() / len(df) * 100,
        'near_zero_pct': (abs(df[current_col]) < 1).sum() / len(df) * 100
    }
    
    print(f"    Current statistics:")
    print(f"      Range: {current_stats['min']:.1f} to {current_stats['max']:.1f} A")
    print(f"      Mean: {current_stats['mean']:.1f} A")
    print(f"      Negative: {current_stats['negative_pct']:.1f}%")
    print(f"      Positive: {current_stats['positive_pct']:.1f}%")
    print(f"      Near zero: {current_stats['near_zero_pct']:.1f}%")
    
    # Determine sign convention
    if current_stats['negative_pct'] > current_stats['positive_pct']:
        print(f"    ✓ Charging current appears to be NEGATIVE")
        return 'negative_charging'
    else:
        print(f"    ⚠️ Charging current appears to be POSITIVE")
        return 'positive_charging'

def load_cleaned_data(vehicle_num):
    """Load cleaned vehicle data with analysis"""
    file_path = CLEANED_PATH / f"vehicle{vehicle_num}_cleaned.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded cleaned Vehicle #{vehicle_num}: {len(df)} rows, {len(df.columns)} columns")
        
        # Analyze data characteristics
        time_unit = analyze_time_interval(df, vehicle_num)
        current_sign = analyze_current_sign(df, vehicle_num)
        
        return df, time_unit, current_sign
    except Exception as e:
        print(f"Error loading Vehicle #{vehicle_num}: {e}")
        return None, None, None

def detect_charging_events(df, vehicle_num, current_sign_convention):
    """
    Detect charging events with proper filtering
    """
    print(f"  Detecting charging events for Vehicle #{vehicle_num}...")
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Identify key columns
    time_col = 'time'
    current_col = 'hv_current'
    soc_col = 'bcell_soc'
    speed_col = 'vhc_speed'
    charging_signal_col = 'charging_signal'
    
    # Strategy 1: Use charging signal (primary method)
    charging_mask = pd.Series(False, index=df.index)
    
    if charging_signal_col in df.columns:
        # Most vehicles use 1 for charging, 3 for driving
        charging_mask = df[charging_signal_col] == 1
        print(f"    Using charging signal: {charging_mask.sum():,} charging points")
    
    # Strategy 2: Current and stationary
    if speed_col in df.columns and current_col in df.columns:
        stationary_mask = df[speed_col] < 1.0
        
        # Adjust current sign based on analysis
        if current_sign_convention == 'positive_charging':
            current_charging_mask = df[current_col] > 1.0  # Positive current for charging
        else:
            current_charging_mask = df[current_col] < -1.0  # Negative current for charging
        
        combined_mask = stationary_mask & current_charging_mask
        
        if 'charging_mask' in locals():
            charging_mask = charging_mask | combined_mask
        else:
            charging_mask = combined_mask
        
        print(f"    Using current/speed: {combined_mask.sum():,} charging points")
    
    # Group consecutive charging points
    charging_groups = (charging_mask != charging_mask.shift()).cumsum()
    
    charging_events = []
    
    for group_id, group_data in df.groupby(charging_groups):
        if charging_mask.iloc[group_data.index[0]]:
            if len(group_data) >= 10:  # Minimum 10 samples
                event = {
                    'vehicle': vehicle_num,
                    'segment_id': group_id,
                    'start_idx': group_data.index[0],
                    'end_idx': group_data.index[-1],
                    'start_time': group_data.iloc[0][time_col],
                    'end_time': group_data.iloc[-1][time_col],
                    'duration_seconds': group_data.iloc[-1][time_col] - group_data.iloc[0][time_col],
                    'start_soc': group_data.iloc[0][soc_col],
                    'end_soc': group_data.iloc[-1][soc_col],
                    'delta_soc': group_data.iloc[-1][soc_col] - group_data.iloc[0][soc_col],
                    'avg_current': group_data[current_col].mean(),
                    'num_samples': len(group_data)
                }
                
                # Basic validation
                is_valid = (
                    event['delta_soc'] >= 5 and  # Minimum ΔSOC
                    event['duration_seconds'] > 300 and  # Minimum 5 minutes
                    0 <= event['start_soc'] <= 100 and
                    0 <= event['end_soc'] <= 100
                )
                
                if is_valid:
                    event['rejection_reason'] = 'Accepted'
                    charging_events.append(event)
                else:
                    event['rejection_reason'] = 'Failed validation'
    
    events_df = pd.DataFrame(charging_events)
    
    if len(events_df) > 0:
        print(f"    Detected {len(events_df)} charging events")
        print(f"    Valid events: {(events_df['rejection_reason'] == 'Accepted').sum()}")
    else:
        print(f"    No charging events detected")
    
    return events_df

def calculate_capacity_with_debug(events_df, df, vehicle_num, time_unit):
    """
    Calculate battery capacity with debugging
    """
    print(f"  Calculating capacity estimates for Vehicle #{vehicle_num}...")
    
    if len(events_df) == 0:
        return pd.DataFrame()
    
    # Filter only accepted events
    valid_events = events_df[events_df['rejection_reason'] == 'Accepted'].copy()
    
    if len(valid_events) == 0:
        print(f"    No valid charging events for capacity calculation")
        return pd.DataFrame()
    
    capacity_estimates = []
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    time_col = 'time'
    current_col = 'hv_current'
    
    print(f"    DEBUG - Analyzing first 5 charging events:")
    
    for idx, event in valid_events.head(5).iterrows():
        # Extract segment
        segment = df.iloc[event['start_idx']:event['end_idx']+1].copy()
        
        # Ensure numeric
        segment[time_col] = pd.to_numeric(segment[time_col], errors='coerce')
        segment[current_col] = pd.to_numeric(segment[current_col], errors='coerce')
        
        # Adjust time based on unit analysis
        if time_unit == 'tenths_of_seconds':
            # Convert to seconds
            time_values = segment[time_col] / 10.0
        elif time_unit == 'minutes':
            # Convert to seconds
            time_values = segment[time_col] * 60.0
        else:
            # Assume seconds
            time_values = segment[time_col]
        
        # Time differences in HOURS
        time_seconds = np.diff(time_values)
        time_hours = time_seconds / 3600.0
        
        # Use absolute current
        current_abs = np.abs(segment[current_col].values)
        
        # Trapezoidal integration
        integrated_current = 0
        for i in range(len(time_hours)):
            avg_current = (current_abs[i] + current_abs[i+1]) / 2
            integrated_current += avg_current * time_hours[i]
        
        # Calculate capacity
        delta_soc_fraction = event['delta_soc'] / 100.0
        estimated_capacity = integrated_current / delta_soc_fraction if delta_soc_fraction > 0 else np.nan
        
        # Calculate SOH
        initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
        soh = (estimated_capacity / initial_capacity) * 100.0 if not np.isnan(estimated_capacity) else np.nan
        
        # Print debug info
        print(f"      Event {event['segment_id']}:")
        print(f"        ΔSOC: {event['delta_soc']:.1f}%")
        print(f"        Duration: {event['duration_seconds']:.0f}s = {event['duration_seconds']/3600:.2f}h")
        print(f"        ∫|I|·dt: {integrated_current:.2f} Ah")
        print(f"        Estimated Capacity: {estimated_capacity:.1f} Ah")
        print(f"        Initial Capacity: {initial_capacity} Ah")
        print(f"        SOH: {soh:.1f}%")
        
        capacity_estimate = {
            'vehicle': vehicle_num,
            'event_id': event['segment_id'],
            'duration_seconds': event['duration_seconds'],
            'duration_hours': event['duration_seconds'] / 3600,
            'delta_soc': event['delta_soc'],
            'integrated_current_ah': integrated_current,
            'estimated_capacity_ah': estimated_capacity,
            'initial_capacity_ah': initial_capacity,
            'soh_percent': soh,
            'avg_current_a': event['avg_current']
        }
        
        capacity_estimates.append(capacity_estimate)
    
    estimates_df = pd.DataFrame(capacity_estimates)
    
    if len(estimates_df) > 0:
        print(f"    Calculated {len(estimates_df)} capacity estimates")
        print(f"    SOH range: {estimates_df['soh_percent'].min():.1f}% to {estimates_df['soh_percent'].max():.1f}%")
    
    return estimates_df

def apply_reasonable_filters(estimates_df, vehicle_num):
    """
    Apply REASONABLE plausibility filters
    """
    print(f"  Applying reasonable filters for Vehicle #{vehicle_num}...")
    
    if len(estimates_df) == 0:
        return pd.DataFrame()
    
    initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
    
    # Create a copy
    filtered_df = estimates_df.copy()
    
    # Remove NaN
    initial_count = len(filtered_df)
    filtered_df = filtered_df.dropna(subset=['estimated_capacity_ah', 'soh_percent'])
    nan_removed = initial_count - len(filtered_df)
    
    # REASONABLE FILTERS (much wider):
    # Capacity: 30% to 200% of initial
    capacity_mask = (filtered_df['estimated_capacity_ah'] >= initial_capacity * 0.3) & \
                    (filtered_df['estimated_capacity_ah'] <= initial_capacity * 2.0)
    
    # SOH: 30% to 200%
    soh_mask = (filtered_df['soh_percent'] >= 30) & (filtered_df['soh_percent'] <= 200)
    
    # ΔSOC: 5% to 90%
    dsoc_mask = (filtered_df['delta_soc'] >= 5) & (filtered_df['delta_soc'] <= 90)
    
    # Duration: 5 min to 24 hours
    duration_mask = (filtered_df['duration_hours'] >= 0.0833) & (filtered_df['duration_hours'] <= 24)
    
    # Combine
    combined_mask = capacity_mask & soh_mask & dsoc_mask & duration_mask
    
    filtered_before = len(filtered_df)
    filtered_df = filtered_df[combined_mask].copy()
    filtered_after = len(filtered_df)
    
    print(f"    Removed {nan_removed} NaN values")
    print(f"    Removed {filtered_before - filtered_after} implausible estimates")
    print(f"    Remaining estimates: {filtered_after}")
    
    return filtered_df

def save_all_estimates_for_analysis(estimates_df, vehicle_num):
    """Save ALL estimates (even filtered) for analysis"""
    if len(estimates_df) == 0:
        return
    
    # Save raw estimates
    raw_file = SOU_LABEL_PATH / f'vehicle{vehicle_num}_raw_estimates.csv'
    estimates_df.to_csv(raw_file, index=False)
    print(f"  ✓ Saved RAW estimates for analysis: {raw_file}")
    
    # Print statistics
    print(f"  Raw SOH statistics:")
    print(f"    Min: {estimates_df['soh_percent'].min():.1f}%")
    print(f"    Max: {estimates_df['soh_percent'].max():.1f}%")
    print(f"    Mean: {estimates_df['soh_percent'].mean():.1f}%")
    print(f"    Median: {estimates_df['soh_percent'].median():.1f}%")
    
    # Check if values are extremely high or low
    if estimates_df['soh_percent'].max() > 1000:
        print(f"  ⚠️ WARNING: Extremely high SOH values (>1000%) detected!")
        print(f"    Likely time unit or current sign issue")
    elif estimates_df['soh_percent'].min() < 1:
        print(f"  ⚠️ WARNING: Extremely low SOH values (<1%) detected!")
        print(f"    Likely time unit or integration issue")

def main():
    """Main function - DEBUG VERSION"""
    print("=" * 70)
    print("PHASE 2: SOH LABEL GENERATION - DEBUG VERSION")
    print("=" * 70)
    print("This version will:")
    print("1. Analyze time units and current sign")
    print("2. Show debug output for first 5 charging events")
    print("3. Save ALL estimates (even unreasonable)")
    print("4. Use wider filters to see distribution")
    print(f"Output will be saved to: {SOU_LABEL_PATH}")
    print()
    
    vehicles_to_process = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    
    for vehicle_num in vehicles_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESSING VEHICLE #{vehicle_num}")
        print(f"{'='*60}")
        
        # Step 1: Load and analyze data
        df, time_unit, current_sign = load_cleaned_data(vehicle_num)
        if df is None:
            continue
        
        # Step 2: Detect charging events
        charging_events = detect_charging_events(df, vehicle_num, current_sign)
        
        if len(charging_events) == 0:
            print(f"  ⚠️ No charging events detected")
            continue
        
        # Step 3: Calculate capacity with DEBUG output
        capacity_estimates = calculate_capacity_with_debug(charging_events, df, vehicle_num, time_unit)
        
        if len(capacity_estimates) == 0:
            print(f"  ⚠️ No capacity estimates calculated")
            continue
        
        # Step 4: Save ALL estimates for analysis
        save_all_estimates_for_analysis(capacity_estimates, vehicle_num)
        
        # Step 5: Apply REASONABLE filters
        filtered_estimates = apply_reasonable_filters(capacity_estimates, vehicle_num)
        
        if len(filtered_estimates) == 0:
            print(f"  ⚠️ All estimates filtered out even with wide filters")
            print(f"  Check the raw_estimates.csv file for actual values")
            continue
        
        print(f"\n  ✅ Vehicle #{vehicle_num} processed successfully!")
        print(f"    Valid SOH estimates: {len(filtered_estimates)}")
        print(f"    SOH range: {filtered_estimates['soh_percent'].min():.1f}% to {filtered_estimates['soh_percent'].max():.1f}%")
        print(f"    Average SOH: {filtered_estimates['soh_percent'].mean():.1f}%")
        
        # Save filtered estimates
        filtered_file = SOU_LABEL_PATH / f'vehicle{vehicle_num}_filtered_estimates.csv'
        filtered_estimates.to_csv(filtered_file, index=False)
        print(f"  ✓ Saved filtered estimates: {filtered_file}")
    
    print(f"\n{'='*70}")
    print("DEBUG COMPLETE")
    print(f"{'='*70}")
    print("\nNEXT STEPS:")
    print("1. Check the raw_estimates.csv files for each vehicle")
    print("2. Look at the SOH values in the debug output")
    print("3. Based on the values, we'll know:")
    print("   - If time units are wrong (SOH >> 100% or << 100%)")
    print("   - If current sign is wrong")
    print("   - What realistic SOH ranges are")
    print("\nCommon patterns and fixes:")
    print("  - SOH ~0.1%: Time in milliseconds (divide by 1000)")
    print("  - SOH ~10,000%: Time in minutes (multiply by 60)")
    print("  - SOH negative: Current sign wrong")
    print("\nOnce we see the debug output, we'll adjust the code accordingly.")

if __name__ == "__main__":
    main()