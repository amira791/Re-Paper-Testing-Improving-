# file: phase1_data_cleaning.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(B)")
CLEANED_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(A)")
CLEANED_PATH.mkdir(exist_ok=True, parents=True)

# Initial capacities from README (for reference)
INITIAL_CAPACITIES = {
    1: 150,  # Ah
    2: 150,
    3: 160,
    4: 160,
    5: 160,
    6: 160,
    7: 120,  # Will be excluded
    8: 645,
    9: 505,
    10: 505
}

def load_vehicle_data(vehicle_num):
    """Load vehicle data"""
    file_path = DATA_PATH / f"vehicle#{vehicle_num}.xlsx"
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded Vehicle #{vehicle_num}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Vehicle #{vehicle_num}: {e}")
        return None

def clean_time_column(df, vehicle_num):
    """
    Fix the time column - most critical issue
    The 65535 values break all time-based calculations
    """
    print(f"  Cleaning time column for Vehicle #{vehicle_num}...")
    
    # Find time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    
    if not time_col:
        print(f"    ERROR: No time column found in Vehicle #{vehicle_num}")
        return df
    
    # Convert to numeric
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    
    # Check for 65535 sentinel values
    sentinel_mask = df[time_col] == 65535
    sentinel_count = sentinel_mask.sum()
    
    if sentinel_count > 0:
        print(f"    Found {sentinel_count:,} 65535 sentinel values in time column")
        
        # Mark sentinel values as NaN
        df.loc[sentinel_mask, time_col] = np.nan
        
        # Count consecutive NaNs to identify gaps
        is_na = df[time_col].isna()
        na_groups = (is_na != is_na.shift()).cumsum()
        
        gap_info = {}
        for group_id, group_data in df.groupby(na_groups):
            if is_na.iloc[group_data.index[0]]:
                gap_length = len(group_data)
                gap_info[gap_length] = gap_info.get(gap_length, 0) + 1
        
        print(f"    Time gaps: {gap_info}")
    
    # Interpolate missing time values
    if df[time_col].isna().any():
        print(f"    Interpolating {df[time_col].isna().sum():,} missing time values...")
        
        # Try to reconstruct time based on 0.1 Hz sampling (10-second intervals)
        if df[time_col].notna().sum() > 0:
            # Forward fill first, then interpolate
            df[time_col] = df[time_col].interpolate(method='linear', limit_direction='both')
            
            # If still NaN at edges, use backward/forward fill
            df[time_col] = df[time_col].fillna(method='ffill').fillna(method='bfill')
    
    # Convert to proper datetime (if Unix timestamp)
    # Check if time values look like Unix timestamps
    sample_time = df[time_col].iloc[0] if len(df) > 0 else 0
    
    if sample_time > 1e9:  # Likely Unix timestamp (seconds since 1970)
        print(f"    Converting Unix timestamps to datetime...")
        df['datetime'] = pd.to_datetime(df[time_col], unit='s')
    elif sample_time > 1e12:  # Likely milliseconds
        print(f"    Converting millisecond timestamps to datetime...")
        df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
    else:
        print(f"    Time format appears to be seconds/minutes from start")
        # Create relative time in seconds
        df['seconds_from_start'] = df[time_col] - df[time_col].min()
    
    # Verify time monotonicity
    if df[time_col].is_monotonic_increasing:
        print(f"    ✓ Time column is monotonic")
    else:
        print(f"    ⚠️ Time column is NOT monotonic - sorting...")
        df = df.sort_values(time_col)
        df = df.reset_index(drop=True)
    
    return df

def handle_sentinel_values(df, vehicle_num):
    """
    Replace all 65535 sentinel values with NaN
    """
    print(f"  Handling sentinel values for Vehicle #{vehicle_num}...")
    
    sentinel_stats = {}
    
    # Check all numeric columns for 65535
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == 'time':  # Already handled
            continue
            
        # Count 65535 values
        sentinel_mask = df[col] == 65535
        sentinel_count = sentinel_mask.sum()
        
        if sentinel_count > 0:
            sentinel_stats[col] = sentinel_count
            
            # Replace with NaN
            df.loc[sentinel_mask, col] = np.nan
            
            # For voltage/temperature columns, also check for 0 or extreme values
            if 'voltage' in col.lower() or 'temp' in col.lower():
                # Additional cleaning for unrealistic values
                voltage_mask = False
                if 'voltage' in col.lower():
                    voltage_mask = (df[col] < 2.5) | (df[col] > 4.5)  # Reasonable cell voltage range
                elif 'temp' in col.lower():
                    voltage_mask = (df[col] < -20) | (df[col] > 60)  # Reasonable temperature range
                
                extreme_count = voltage_mask.sum()
                if extreme_count > 0:
                    df.loc[voltage_mask, col] = np.nan
                    sentinel_stats[f'{col}_extreme'] = extreme_count
    
    if sentinel_stats:
        print(f"    Found sentinel values in {len(sentinel_stats)} columns:")
        for col, count in sentinel_stats.items():
            print(f"      - {col}: {count:,} values replaced with NaN")
    
    return df, sentinel_stats

def fix_mileage_data(df, vehicle_num):
    """
    Fix mileage data (vhc_totalMile column)
    """
    print(f"  Fixing mileage data for Vehicle #{vehicle_num}...")
    
    # Find mileage column
    mileage_col = None
    for col in df.columns:
        if 'mile' in col.lower():
            mileage_col = col
            break
    
    if not mileage_col:
        print(f"    No mileage column found")
        return df
    
    # Convert to numeric
    df[mileage_col] = pd.to_numeric(df[mileage_col], errors='coerce')
    
    # Check for 65535 values (already handled, but check again)
    sentinel_mask = df[mileage_col] == 65535
    if sentinel_mask.any():
        df.loc[sentinel_mask, mileage_col] = np.nan
    
    # Strategy 1: Use speed to estimate mileage where missing
    speed_col = None
    for col in df.columns:
        if 'speed' in col.lower():
            speed_col = col
            break
    
    if speed_col and df[mileage_col].isna().any():
        print(f"    Estimating missing mileage from speed data...")
        
        # Create a copy for estimation
        mileage_est = df[mileage_col].copy()
        
        # Fill known values first
        known_mask = mileage_est.notna()
        
        if known_mask.any():
            # Forward fill known values
            mileage_est = mileage_est.ffill()
            
            # Calculate incremental distance from speed
            time_col = [col for col in df.columns if 'time' in col.lower()][0]
            
            # Calculate time difference in hours
            time_sec = df[time_col].diff().fillna(0)
            time_hours = time_sec / 3600
            
            # Distance = speed * time
            incremental_distance = df[speed_col] * time_hours
            
            # Add incremental distance to last known mileage
            mileage_cumulative = incremental_distance.cumsum()
            
            # Start from first known mileage
            if known_mask.iloc[0]:
                start_mileage = mileage_est.iloc[0]
                mileage_est = start_mileage + mileage_cumulative - mileage_cumulative.iloc[0]
            else:
                # If no starting mileage, estimate from average
                mileage_est = mileage_cumulative
        
        df[mileage_col] = mileage_est
    
    # Strategy 2: Simple interpolation for remaining gaps
    if df[mileage_col].isna().any():
        print(f"    Interpolating remaining mileage gaps...")
        df[mileage_col] = df[mileage_col].interpolate(method='linear', limit_direction='both')
        
        # If still NaN at edges, use nearest
        df[mileage_col] = df[mileage_col].ffill().bfill()
    
    # Verify mileage is non-decreasing
    mileage_decrease = (df[mileage_col].diff() < -0.1).sum()  # Allow small negative due to noise
    if mileage_decrease > 0:
        print(f"    ⚠️ Found {mileage_decrease} mileage decreases - correcting...")
        # Use cumulative max to ensure non-decreasing
        df[mileage_col] = df[mileage_col].cummax()
    
    print(f"    Final mileage range: {df[mileage_col].min():.1f} to {df[mileage_col].max():.1f} km")
    
    return df

def validate_soc_data(df, vehicle_num):
    """
    Validate and clean SOC data
    """
    print(f"  Validating SOC data for Vehicle #{vehicle_num}...")
    
    # Find SOC column
    soc_col = None
    for col in df.columns:
        if 'soc' in col.lower():
            soc_col = col
            break
    
    if not soc_col:
        print(f"    No SOC column found")
        return df, {}
    
    validation_stats = {
        'total_rows': len(df),
        'initial_nan': df[soc_col].isna().sum(),
        'out_of_range': 0,
        'large_jumps': 0,
        'corrected_values': 0
    }
    
    # Convert to numeric
    df[soc_col] = pd.to_numeric(df[soc_col], errors='coerce')
    
    # 1. Remove SOC values outside 0-100% range
    out_of_range_mask = (df[soc_col] < 0) | (df[soc_col] > 100)
    validation_stats['out_of_range'] = out_of_range_mask.sum()
    
    if out_of_range_mask.any():
        print(f"    Found {out_of_range_mask.sum():,} SOC values outside 0-100% range")
        df.loc[out_of_range_mask, soc_col] = np.nan
    
    # 2. Check for unrealistic jumps (>10% in 10 seconds)
    time_col = [col for col in df.columns if 'time' in col.lower()][0]
    
    # Calculate SOC rate of change per second
    soc_diff = df[soc_col].diff().abs()
    time_diff = df[time_col].diff().fillna(10)  # Assume 10 seconds if unknown
    
    # SOC change per second
    soc_rate = soc_diff / time_diff
    
    # Flag unrealistic jumps (>1% per second)
    jump_mask = soc_rate > 1.0  # More than 1% SOC change per second
    validation_stats['large_jumps'] = jump_mask.sum()
    
    if jump_mask.any():
        print(f"    Found {jump_mask.sum():,} unrealistic SOC jumps (>1%/sec)")
        
        # Mark these as NaN for interpolation
        df.loc[jump_mask, soc_col] = np.nan
    
    # 3. Fill missing SOC values (after cleaning)
    initial_nan = df[soc_col].isna().sum()
    if initial_nan > 0:
        print(f"    Filling {initial_nan:,} missing SOC values...")
        
        # Interpolate with limit
        df[soc_col] = df[soc_col].interpolate(method='linear', limit=10, limit_direction='both')
        
        # Forward/backward fill for edges
        df[soc_col] = df[soc_col].ffill().bfill()
        
        validation_stats['corrected_values'] = initial_nan - df[soc_col].isna().sum()
    
    # 4. Final sanity check
    final_nan = df[soc_col].isna().sum()
    if final_nan > 0:
        print(f"    ⚠️ WARNING: Still {final_nan:,} NaN values in SOC after cleaning")
    
    soc_range = df[soc_col].min(), df[soc_col].max()
    print(f"    Final SOC range: {soc_range[0]:.1f}% to {soc_range[1]:.1f}%")
    
    validation_stats['final_nan'] = final_nan
    validation_stats['soc_range'] = soc_range
    
    return df, validation_stats

def clean_numeric_columns(df, vehicle_num):
    """
    Clean all numeric columns (general cleaning)
    """
    print(f"  Cleaning all numeric columns for Vehicle #{vehicle_num}...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Skip already handled columns
        if col in ['time', 'datetime', 'seconds_from_start']:
            continue
        
        # Convert to numeric (safety check)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove extreme outliers (beyond 5 standard deviations)
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        if col_std > 0:  # Avoid division by zero
            z_score = np.abs((df[col] - col_mean) / col_std)
            outlier_mask = z_score > 5
            
            if outlier_mask.any():
                outlier_count = outlier_mask.sum()
                # Replace extreme outliers with NaN for interpolation
                df.loc[outlier_mask, col] = np.nan
    
    # Forward fill for slow-changing variables
    slow_cols = [col for col in df.columns if any(x in col.lower() for x in ['mile', 'soc', 'temp'])]
    for col in slow_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    return df

def create_cleaning_report(vehicle_num, time_stats, sentinel_stats, mileage_info, soc_stats):
    """Create a cleaning report for each vehicle"""
    report = {
        'vehicle': vehicle_num,
        'time_sentinel_fixed': time_stats.get('sentinel_count', 0),
        'time_gaps': time_stats.get('gap_info', {}),
        'sentinel_columns': len(sentinel_stats),
        'sentinel_counts': sentinel_stats,
        'mileage_range': mileage_info.get('range', (0, 0)),
        'soc_initial_nan': soc_stats.get('initial_nan', 0),
        'soc_out_of_range': soc_stats.get('out_of_range', 0),
        'soc_large_jumps': soc_stats.get('large_jumps', 0),
        'soc_corrected': soc_stats.get('corrected_values', 0),
        'soc_final_nan': soc_stats.get('final_nan', 0),
        'soc_range': soc_stats.get('soc_range', (0, 0))
    }
    
    return report

def plot_cleaning_results(df, vehicle_num, original_df):
    """Create visualization of cleaning results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time column before/after
    ax1 = axes[0, 0]
    time_col = [col for col in df.columns if 'time' in col.lower()][0]
    
    ax1.plot(original_df.index, original_df[time_col], 'r.', alpha=0.3, label='Original (with 65535)', markersize=1)
    ax1.plot(df.index, df[time_col], 'b-', alpha=0.7, label='Cleaned', linewidth=0.5)
    ax1.set_title(f'Vehicle #{vehicle_num} - Time Column Cleaning')
    ax1.set_xlabel('Row Index')
    ax1.set_ylabel('Time Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SOC before/after
    ax2 = axes[0, 1]
    soc_col = [col for col in df.columns if 'soc' in col.lower()][0]
    
    if soc_col in original_df.columns and soc_col in df.columns:
        ax2.plot(original_df.index, original_df[soc_col], 'r.', alpha=0.3, label='Original', markersize=1)
        ax2.plot(df.index, df[soc_col], 'b-', alpha=0.7, label='Cleaned', linewidth=0.5)
        ax2.set_title(f'Vehicle #{vehicle_num} - SOC Cleaning')
        ax2.set_xlabel('Row Index')
        ax2.set_ylabel('SOC (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mileage before/after
    ax3 = axes[1, 0]
    mileage_col = None
    for col in df.columns:
        if 'mile' in col.lower():
            mileage_col = col
            break
    
    if mileage_col and mileage_col in original_df.columns and mileage_col in df.columns:
        ax3.plot(original_df.index, original_df[mileage_col], 'r.', alpha=0.3, label='Original', markersize=1)
        ax3.plot(df.index, df[mileage_col], 'b-', alpha=0.7, label='Cleaned', linewidth=0.5)
        ax3.set_title(f'Vehicle #{vehicle_num} - Mileage Cleaning')
        ax3.set_xlabel('Row Index')
        ax3.set_ylabel('Mileage (km)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Voltage data example
    ax4 = axes[1, 1]
    voltage_cols = [col for col in df.columns if 'voltage' in col.lower() and 'max' in col.lower()]
    
    if voltage_cols and voltage_cols[0] in original_df.columns and voltage_cols[0] in df.columns:
        voltage_col = voltage_cols[0]
        # Take a sample for clarity
        sample_idx = np.linspace(0, len(df)-1, min(1000, len(df))).astype(int)
        
        ax4.plot(original_df.index[sample_idx], original_df[voltage_col].iloc[sample_idx], 
                'r.', alpha=0.3, label='Original', markersize=2)
        ax4.plot(df.index[sample_idx], df[voltage_col].iloc[sample_idx], 
                'b.', alpha=0.7, label='Cleaned', markersize=2)
        ax4.set_title(f'Vehicle #{vehicle_num} - Max Cell Voltage Cleaning')
        ax4.set_xlabel('Row Index')
        ax4.set_ylabel('Voltage (V)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CLEANED_PATH / f'vehicle{vehicle_num}_cleaning_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main cleaning function"""
    print("=" * 70)
    print("PHASE 1: DATA CLEANING")
    print("=" * 70)
    print("Fixing critical issues: Time column, 65535 sentinel values, Mileage, SOC")
    print(f"Output will be saved to: {CLEANED_PATH}")
    print()
    
    all_reports = []
    
    # Process vehicles 1-10, but skip Vehicle #7
    vehicles_to_process = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    
    for vehicle_num in vehicles_to_process:
        print(f"\n{'='*60}")
        print(f"CLEANING VEHICLE #{vehicle_num}")
        print(f"{'='*60}")
        
        # Load original data
        original_df = load_vehicle_data(vehicle_num)
        if original_df is None:
            continue
        
        # Make a copy for cleaning
        df = original_df.copy()
        
        # Step 1: Fix Time Column (MOST CRITICAL)
        df = clean_time_column(df, vehicle_num)
        
        # Step 2: Handle Sentinel Values (65535)
        df, sentinel_stats = handle_sentinel_values(df, vehicle_num)
        
        # Step 3: Fix Mileage Data
        df = fix_mileage_data(df, vehicle_num)
        
        # Get mileage info for report
        mileage_col = None
        for col in df.columns:
            if 'mile' in col.lower():
                mileage_col = col
                break
        
        mileage_info = {}
        if mileage_col:
            mileage_info['range'] = (df[mileage_col].min(), df[mileage_col].max())
        
        # Step 4: SOC Data Validation
        df, soc_stats = validate_soc_data(df, vehicle_num)
        
        # Step 5: General numeric column cleaning
        df = clean_numeric_columns(df, vehicle_num)
        
        # Create cleaning report
        time_stats = {
            'sentinel_count': sentinel_stats.get('time', 0) if isinstance(sentinel_stats, dict) else 0,
            'gap_info': {}  # Could add gap analysis here
        }
        
        report = create_cleaning_report(vehicle_num, time_stats, sentinel_stats, 
                                       mileage_info, soc_stats)
        all_reports.append(report)
        
        # Save cleaned data
        cleaned_file = CLEANED_PATH / f'vehicle{vehicle_num}_cleaned.csv'
        df.to_csv(cleaned_file, index=False)
        print(f"  ✓ Saved cleaned data: {cleaned_file}")
        
        # Create visualization
        plot_cleaning_results(df, vehicle_num, original_df)
        print(f"  ✓ Created cleaning visualization")
        
        # Print summary
        print(f"\n  Cleaning Summary for Vehicle #{vehicle_num}:")
        print(f"    - Time sentinel values fixed: {report['time_sentinel_fixed']:,}")
        print(f"    - Sentinel columns cleaned: {report['sentinel_columns']}")
        print(f"    - SOC range: {report['soc_range'][0]:.1f}% to {report['soc_range'][1]:.1f}%")
        print(f"    - Mileage range: {report['mileage_range'][0]:.1f} to {report['mileage_range'][1]:.1f} km")
    
    # Create master cleaning report
    if all_reports:
        reports_df = pd.DataFrame(all_reports)
        reports_df.to_csv(CLEANED_PATH / 'cleaning_summary_report.csv', index=False)
        
        print(f"\n{'='*70}")
        print("CLEANING COMPLETE - SUMMARY")
        print(f"{'='*70}")
        
        # Print overall statistics
        print(f"\nVehicles processed: {len(reports_df)}")
        print(f"Vehicles excluded: Vehicle #7 (different structure)")
        
        # Summary statistics
        total_sentinel = reports_df['time_sentinel_fixed'].sum()
        total_soc_corrected = reports_df['soc_corrected'].sum()
        
        print(f"\nTotal sentinel values (65535) removed: {total_sentinel:,}")
        print(f"Total SOC values corrected: {total_soc_corrected:,}")
        
        # Check for remaining issues
        remaining_nan = reports_df['soc_final_nan'].sum()
        if remaining_nan > 0:
            print(f"⚠️ WARNING: {remaining_nan:,} NaN values remain in SOC data")
        
        print(f"\nCleaned data saved to: {CLEANED_PATH}")
        print("Files created for each vehicle:")
        print("  - vehicleX_cleaned.csv (cleaned data)")
        print("  - vehicleX_cleaning_results.png (visualization)")
        print("  - cleaning_summary_report.csv (summary statistics)")
        
        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print("1. Verify the cleaned data looks reasonable")
        print("2. Run Phase 2: SOH Label Generation on cleaned data")
        print("3. Check that SOH values are now physically plausible")
        print(f"{'='*70}")
    
    else:
        print("No vehicles were successfully processed.")

if __name__ == "__main__":
    main()