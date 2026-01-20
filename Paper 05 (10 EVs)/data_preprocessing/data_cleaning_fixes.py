"""
COMPREHENSIVE DATA CLEANING TO FIX CRITICAL ISSUES
Fixes: Temperature outliers, missing data, time series issues, SOH artifacts
UPDATED VERSION: Fixes missing value increase and vehicle imbalance
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load data and apply comprehensive cleaning"""
    
    # Load data
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    df = pd.read_parquet(data_path / "all_vehicles_combined.parquet")
    
    print("="*80)
    print("STARTING COMPREHENSIVE DATA CLEANING")
    print("="*80)
    print(f"Original shape: {df.shape}")
    
    # Track cleaning operations
    cleaning_log = []
    
    # FIX 1: Handle Vehicle 7 temperature data
    cleaning_log.append(f"FIX 1: Handling Vehicle 7 temperature data")
    if 'bcell_Temp' in df.columns and 'bcell_maxTemp' in df.columns:
        # Count before
        missing_before = df['bcell_maxTemp'].isna().sum()
        
        # Fill Vehicle 7 missing temps with single temp value
        mask = df['bcell_maxTemp'].isna() & df['bcell_Temp'].notna()
        df.loc[mask, 'bcell_maxTemp'] = df.loc[mask, 'bcell_Temp']
        df.loc[mask, 'bcell_minTemp'] = df.loc[mask, 'bcell_Temp']
        
        # Also fill bcell_minTemp if it exists and has NaN
        if 'bcell_minTemp' in df.columns:
            mask = df['bcell_minTemp'].isna() & df['bcell_Temp'].notna()
            df.loc[mask, 'bcell_minTemp'] = df.loc[mask, 'bcell_Temp']
        
        missing_after = df['bcell_maxTemp'].isna().sum()
        cleaning_log.append(f"   Fixed {missing_before - missing_after:,} missing temperature values")
    
    # FIX 2: Fix temperature outliers
    cleaning_log.append(f"\nFIX 2: Fixing temperature outliers")
    
    # Realistic EV battery temperature range: -20°C to 60°C (with safety margin)
    temp_cols = ['bcell_maxTemp', 'bcell_minTemp', 'bcell_Temp']
    
    for col in temp_cols:
        if col in df.columns:
            outliers_before = ((df[col] < -20) | (df[col] > 60)).sum()
            df[col] = df[col].clip(-20, 60)
                
            outliers_after = ((df[col] < -20) | (df[col] > 60)).sum()
            if outliers_before > 0:
                cleaning_log.append(f"   {col}: Fixed {outliers_before:,} outliers")
    
    # FIX 3: Fix SOH clipping artifacts
    cleaning_log.append(f"\nFIX 3: Fixing SOH clipping artifacts")
    
    if 'soh_capacity' in df.columns:
        # Count values at boundaries
        at_70 = (df['soh_capacity'] == 70).sum()
        at_105 = (df['soh_capacity'] == 105).sum()
        total_boundary = at_70 + at_105
        
        cleaning_log.append(f"   Found {at_70:,} rows at 70% SOH (lower bound)")
        cleaning_log.append(f"   Found {at_105:,} rows at 105% SOH (upper bound)")
        cleaning_log.append(f"   Total at boundaries: {total_boundary:,} ({total_boundary/len(df)*100:.1f}%)")
        
        # Option 1: Smooth boundaries by adding small noise
        np.random.seed(42)  # For reproducibility
        noise_70 = np.random.uniform(-1.5, 1.5, len(df))
        noise_105 = np.random.uniform(-1.5, 1.5, len(df))
        
        # Apply noise only to boundary values
        mask_70 = df['soh_capacity'] == 70
        mask_105 = df['soh_capacity'] == 105
        
        df.loc[mask_70, 'soh_capacity'] = 70 + noise_70[mask_70]
        df.loc[mask_105, 'soh_capacity'] = 105 + noise_105[mask_105]
        
        # Re-clip to slightly wider bounds (65-110%)
        df['soh_capacity'] = df['soh_capacity'].clip(65, 110)
        
        # Also fix soh_final if it exists
        if 'soh_final' in df.columns:
            df['soh_final'] = df['soh_final'].clip(65, 110)
    
    # FIX 4: Handle missing voltage values (65535 sensor faults) - IMPROVED
    cleaning_log.append(f"\nFIX 4: Handling sensor faults (65535 values)")
    
    voltage_cols = ['bcell_maxVoltage', 'bcell_minVoltage', 'hv_voltage']
    for col in voltage_cols:
        if col in df.columns:
            # Count sensor faults
            faults_before = (df[col] == 65535).sum()
            if faults_before > 0:
                # Store indices of faults for targeted imputation
                fault_indices = df[df[col] == 65535].index
                
                # Replace with NaN
                df.loc[df[col] == 65535, col] = np.nan
                
                # Smart imputation: use interpolation within each vehicle
                df = df.sort_values(['vehicle_id', 'time'])
                
                # Interpolate only at fault locations - FIXED VERSION
                for vehicle_id in df['vehicle_id'].unique():
                    vehicle_mask = df['vehicle_id'] == vehicle_id
                    
                    # Get indices for this vehicle
                    vehicle_indices = df[vehicle_mask].index
                    vehicle_faults = fault_indices.intersection(vehicle_indices)
                    
                    if len(vehicle_faults) > 0:
                        # Get the column data for this vehicle
                        vehicle_data = df.loc[vehicle_mask, col].copy()
                        
                        # Interpolate missing values
                        vehicle_data = vehicle_data.interpolate(method='linear', limit_direction='both', limit=10)
                        
                        # Update only the vehicle's data
                        df.loc[vehicle_mask, col] = vehicle_data
                
                # Final check: fill any remaining with vehicle median
                remaining_nan = df[col].isna().sum()
                if remaining_nan > 0:
                    vehicle_medians = df.groupby('vehicle_id')[col].transform('median')
                    df[col] = df[col].fillna(vehicle_medians)
                
                cleaning_log.append(f"   {col}: Fixed {faults_before:,} sensor faults")
    
    # FIX 5: Fix broken time series and create session-based features
    cleaning_log.append(f"\nFIX 5: Fixing broken time series")
    
    if 'time' in df.columns:
        # Ensure data is sorted
        df = df.sort_values(['vehicle_id', 'time'])
        
        # Identify driving sessions (speed > 1 km/h)
        df['is_driving'] = (df['vhc_speed'] > 1).astype(int)
        
        # Session changes when: vehicle changes OR driving state changes significantly
        df['session_change'] = (
            (df['vehicle_id'] != df['vehicle_id'].shift()) |
            (df['is_driving'].diff().abs() > 0.5)
        ).cumsum()
        
        # Create relative time within each session
        df['relative_time'] = df.groupby('session_change').cumcount()
        
        # Calculate time differences within sessions (ignore negative jumps)
        df['time_diff'] = df.groupby('session_change')['time'].diff()
        df['time_diff'] = df['time_diff'].clip(0, 3600)  # Max 1 hour gap
        
        cleaning_log.append(f"   Created {df['session_change'].nunique():,} driving/charging sessions")
        cleaning_log.append(f"   Created relative time within sessions")
    
    # FIX 6: Balance Vehicle 7 dominance - NOW MODIFIES MAIN DATAFRAME
    cleaning_log.append(f"\nFIX 6: Balancing Vehicle 7 dominance")
    
    if 'vehicle_id' in df.columns:
        vehicle_counts = df['vehicle_id'].value_counts()
        
        cleaning_log.append(f"   Vehicle distribution before balancing:")
        for vid, count in vehicle_counts.items():
            cleaning_log.append(f"     Vehicle {vid}: {count:,} rows")
        
        # Calculate target rows (use 75th percentile instead of 1.5x median)
        target_rows = int(vehicle_counts.quantile(0.75))
        
        # Create balanced dataset
        balanced_dfs = []
        for vid in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vid].copy()
            
            if len(vehicle_data) > target_rows:
                # Stratified sampling by time to preserve temporal patterns
                vehicle_data = vehicle_data.sort_values('time')
                
                # Create time-based segments for sampling
                n_segments = min(5, target_rows // 2000)  # Create up to 5 segments
                
                if n_segments > 1 and len(vehicle_data) > n_segments * 100:
                    # Add segment labels
                    vehicle_data['segment'] = pd.qcut(vehicle_data['time'], q=n_segments, labels=False, duplicates='drop')
                    
                    # Sample proportionally from each segment
                    samples_per_segment = target_rows // n_segments
                    sampled_data = []
                    
                    for segment_num in range(n_segments):
                        segment_data = vehicle_data[vehicle_data['segment'] == segment_num]
                        if len(segment_data) > 0:
                            sample_size = min(samples_per_segment, len(segment_data))
                            sampled_data.append(segment_data.sample(n=sample_size, random_state=42))
                    
                    balanced_data = pd.concat(sampled_data, ignore_index=True)
                    
                    # If we need more rows, take random sample from remainder
                    if len(balanced_data) < target_rows:
                        remaining = vehicle_data[~vehicle_data.index.isin(balanced_data.index)]
                        needed = target_rows - len(balanced_data)
                        if len(remaining) > needed:
                            balanced_data = pd.concat([balanced_data, 
                                                     remaining.sample(n=needed, random_state=42)], 
                                                    ignore_index=True)
                else:
                    # Simple random sample
                    balanced_data = vehicle_data.sample(n=target_rows, random_state=42)
            else:
                # Keep all data for vehicles with less than target
                balanced_data = vehicle_data
            
            balanced_dfs.append(balanced_data)
        
        # REPLACE THE ORIGINAL df WITH THE BALANCED ONE
        df_original_size = len(df)
        df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Log the changes
        vehicle_counts_after = df['vehicle_id'].value_counts()
        imbalance_before = vehicle_counts.max() / vehicle_counts.min()
        imbalance_after = vehicle_counts_after.max() / vehicle_counts_after.min()
        
        cleaning_log.append(f"\n   After balancing:")
        cleaning_log.append(f"     Original dataset: {df_original_size:,} rows")
        cleaning_log.append(f"     Balanced dataset: {len(df):,} rows")
        cleaning_log.append(f"     Target per vehicle: ~{target_rows:,} rows")
        cleaning_log.append(f"     Imbalance ratio: {imbalance_before:.1f}x → {imbalance_after:.1f}x")
        
        # Show new distribution
        for vid in sorted(df['vehicle_id'].unique()):
            count = vehicle_counts_after[vid]
            cleaning_log.append(f"     Vehicle {vid}: {count:,} rows")
    
    # FIX 7: Create useful derived features
    cleaning_log.append(f"\nFIX 7: Creating derived features")
    
    # Cell imbalance
    if all(col in df.columns for col in ['bcell_maxVoltage', 'bcell_minVoltage']):
        df['cell_imbalance'] = df['bcell_maxVoltage'] - df['bcell_minVoltage']
        cleaning_log.append(f"   Created: cell_imbalance")
    
    # Temperature gradient
    if all(col in df.columns for col in ['bcell_maxTemp', 'bcell_minTemp']):
        df['temp_gradient'] = df['bcell_maxTemp'] - df['bcell_minTemp']
        cleaning_log.append(f"   Created: temp_gradient")
    
    # Power calculation
    if all(col in df.columns for col in ['hv_voltage', 'hv_current']):
        df['power_kw'] = (df['hv_voltage'] * df['hv_current']) / 1000
        cleaning_log.append(f"   Created: power_kw")
    
    # SOC-based features
    if 'bcell_soc' in df.columns:
        df['soc_category'] = pd.cut(df['bcell_soc'], 
                                    bins=[0, 20, 40, 60, 80, 100],
                                    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
        cleaning_log.append(f"   Created: soc_category")
    
    # Driving intensity
    if 'vhc_speed' in df.columns:
        df['driving_intensity'] = pd.cut(df['vhc_speed'],
                                        bins=[-1, 0, 20, 60, 200],
                                        labels=['Parked', 'City', 'Highway', 'Extreme'])
        cleaning_log.append(f"   Created: driving_intensity")
    
    # FIX 8: Enhanced missing value imputation (FIXED VERSION)
    cleaning_log.append(f"\nFIX 8: Enhanced missing value imputation")
        
    # List of ALL columns that might have missing values (expanded list)
    impute_cols = ['vhc_speed', 'hv_current', 'hv_voltage', 'bcell_soc', 
                   'bcell_maxVoltage', 'bcell_minVoltage', 'bcell_maxTemp', 
                   'bcell_minTemp', 'bcell_Temp', 'cell_imbalance', 
                   'temp_gradient', 'power_kw', 'soh_capacity', 'soh_final']
    
    # Only include columns that exist in the dataframe
    impute_cols = [col for col in impute_cols if col in df.columns]
    
    # Track overall missing values
    total_missing_before = df[impute_cols].isna().sum().sum()
    cleaning_log.append(f"   Total missing values before imputation: {total_missing_before:,}")
    
    # Ensure data is sorted by vehicle and time for proper interpolation
    if 'time' in df.columns:
        df = df.sort_values(['vehicle_id', 'time'])
    
    for col in impute_cols:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                # METHOD 1: Try interpolation for time-series data first - FIXED
                if col in ['bcell_maxTemp', 'bcell_minTemp', 'bcell_Temp', 
                          'bcell_maxVoltage', 'bcell_minVoltage', 'bcell_soc',
                          'vhc_speed', 'hv_current', 'hv_voltage']:
                    # FIX: Use transform instead of apply to maintain index alignment
                    df[col] = df.groupby('vehicle_id')[col].transform(
                        lambda x: x.interpolate(method='linear', limit_direction='both', limit=20)
                    )
                
                # METHOD 2: Forward/backward fill with limits
                df[col] = df.groupby('vehicle_id')[col].ffill(limit=10)
                df[col] = df.groupby('vehicle_id')[col].bfill(limit=10)
                
                # METHOD 3: Rolling median for remaining values - FIXED
                if df[col].isna().sum() > 0:
                    # Use transform to maintain index alignment
                    df[col] = df.groupby('vehicle_id')[col].transform(
                        lambda x: x.fillna(x.rolling(30, min_periods=1).median())
                    )
                
                # METHOD 4: Vehicle-specific median
                if df[col].isna().sum() > 0:
                    vehicle_medians = df.groupby('vehicle_id')[col].transform('median')
                    df[col] = df[col].fillna(vehicle_medians)
                
                # METHOD 5: Global median as last resort
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
                
                missing_after = df[col].isna().sum()
                reduction = missing_before - missing_after
                if reduction > 0:
                    cleaning_log.append(f"   {col}: Imputed {reduction:,} values ({missing_before:,} → {missing_after:,})")
    
    total_missing_after = df.isnull().sum().sum()
    reduction_pct = ((total_missing_before - total_missing_after) / total_missing_before * 100) if total_missing_before > 0 else 0
    cleaning_log.append(f"   Total reduction: {total_missing_before - total_missing_after:,} values ({reduction_pct:.1f}%)")
    cleaning_log.append(f"   Final missing values: {total_missing_after:,}")
    
    # Save cleaning log
    with open('data_cleaning_log.txt', 'w', encoding='utf-8') as f:
        f.write("DATA CLEANING LOG\n")
        f.write("="*50 + "\n\n")
        for line in cleaning_log:
            f.write(line + "\n")
    
    # Save cleaned datasets
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    # Save full cleaned dataset (now balanced)
    df.to_parquet(data_path / "all_vehicles_cleaned.parquet", index=False)
    
    print("\n" + "="*80)
    print("CLEANING COMPLETE!")
    print("="*80)
    
    # Print summary
    for line in cleaning_log:
        print(line)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Saved to: {data_path / 'all_vehicles_cleaned.parquet'}")
    
    print(f"\nCleaning log saved to: data_cleaning_log.txt")
    
    return df

def diagnose_missing_value_issues(df_before, df_after):
    """Diagnose where missing values are being created"""
    print("\n" + "="*80)
    print("DIAGNOSING MISSING VALUE INCREASE")
    print("="*80)
    
    # Compare column by column
    problematic_cols = []
    
    for col in df_before.columns:
        if col in df_after.columns:
            missing_before = df_before[col].isna().sum()
            missing_after = df_after[col].isna().sum()
            
            if missing_after > missing_before:
                increase = missing_after - missing_before
                increase_pct = (increase / len(df_before)) * 100
                problematic_cols.append((col, missing_before, missing_after, increase, increase_pct))
    
    if problematic_cols:
        print("Columns with INCREASED missing values after cleaning:")
        print(f"{'Column':<25} {'Before':>10} {'After':>10} {'Increase':>10} {'% Increase':>10}")
        print("-" * 70)
        
        for col, before, after, inc, pct in sorted(problematic_cols, key=lambda x: x[3], reverse=True):
            print(f"{col:<25} {before:>10,} {after:>10,} {inc:>10,} {pct:>9.2f}%")
    else:
        print("✅ No columns have increased missing values!")
    
    return problematic_cols

def verify_cleaning(df):
    """Verify that cleaning was successful"""
    print("\n" + "="*80)
    print("VERIFYING DATA CLEANING")
    print("="*80)
    
    verification_log = []
    
    # 1. Check temperature ranges
    temp_cols = [col for col in ['bcell_maxTemp', 'bcell_minTemp', 'bcell_Temp'] if col in df.columns]
    for col in temp_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        verification_log.append(f"{col}: Range = {min_val:.1f}°C to {max_val:.1f}°C")
        
        if min_val < -20 or max_val > 60:
            verification_log.append(f"  ⚠️ WARNING: {col} still has unrealistic values!")
        else:
            verification_log.append(f"  ✅ OK: Within realistic range")
    
    # 2. Check SOH boundaries
    if 'soh_capacity' in df.columns:
        at_70 = (df['soh_capacity'] == 70).sum()
        at_105 = (df['soh_capacity'] == 105).sum()
        verification_log.append(f"\nSOH boundary values:")
        verification_log.append(f"  At exactly 70%: {at_70:,} rows")
        verification_log.append(f"  At exactly 105%: {at_105:,} rows")
        
        if at_70 + at_105 < 1000:  # Arbitrary threshold
            verification_log.append(f"  ✅ OK: Minimal boundary clustering")
        else:
            verification_log.append(f"  ⚠️ WARNING: Still many boundary values")
    
    # 3. Check missing values
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    verification_log.append(f"\nMissing values after cleaning:")
    if len(missing_cols) == 0:
        verification_log.append(f"  ✅ OK: No missing values")
    else:
        verification_log.append(f"  ⚠️ WARNING: {len(missing_cols)} columns still have missing values")
        total_missing = missing_summary.sum()
        verification_log.append(f"  Total missing: {total_missing:,} ({total_missing/len(df)*100:.2f}%)")
        for col, count in missing_cols.head(10).items():  # Show top 10 only
            verification_log.append(f"    {col}: {count:,} missing ({count/len(df)*100:.2f}%)")
    
    # 4. Check time series issues
    if 'time_diff' in df.columns:
        negative_times = (df['time_diff'] < 0).sum()
        large_gaps = (df['time_diff'] > 3600).sum()
        
        verification_log.append(f"\nTime series verification:")
        verification_log.append(f"  Negative time intervals: {negative_times:,}")
        verification_log.append(f"  Gaps > 1 hour: {large_gaps:,}")
        
        if negative_times == 0:
            verification_log.append(f"  ✅ OK: No negative time intervals")
        else:
            verification_log.append(f"  ⚠️ WARNING: Still have negative time intervals")
    
    # 5. Check vehicle distribution
    if 'vehicle_id' in df.columns:
        vehicle_counts = df['vehicle_id'].value_counts()
        imbalance_ratio = vehicle_counts.max() / vehicle_counts.min()
        
        verification_log.append(f"\nVehicle distribution:")
        verification_log.append(f"  Most data: Vehicle {vehicle_counts.idxmax()} ({vehicle_counts.max():,} rows)")
        verification_log.append(f"  Least data: Vehicle {vehicle_counts.idxmin()} ({vehicle_counts.min():,} rows)")
        verification_log.append(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
        
        if imbalance_ratio > 10:
            verification_log.append(f"  ⚠️ WARNING: Severe imbalance (>10x)")
        elif imbalance_ratio > 5:
            verification_log.append(f"  ⚠️ CAUTION: Moderate imbalance (5-10x)")
        else:
            verification_log.append(f"  ✅ OK: Reasonable balance")
        
        # Show all vehicle counts
        verification_log.append(f"  Detailed distribution:")
        for vid in sorted(vehicle_counts.index):
            verification_log.append(f"    Vehicle {vid}: {vehicle_counts[vid]:,} rows")
    
    # 6. Check new features
    new_features = ['cell_imbalance', 'temp_gradient', 'power_kw', 'soc_category', 'driving_intensity']
    verification_log.append(f"\nNew features created:")
    for feature in new_features:
        if feature in df.columns:
            missing = df[feature].isna().sum()
            verification_log.append(f"  {feature}: {'✅ Present' if missing == 0 else f'⚠️ {missing:,} missing'}")
    
    # Print verification log
    for line in verification_log:
        print(line)
    
    # Save verification
    with open('data_cleaning_verification.txt', 'w', encoding='utf-8') as f:
        f.write("DATA CLEANING VERIFICATION\n")
        f.write("="*50 + "\n\n")
        for line in verification_log:
            f.write(line + "\n")
    
    return verification_log

def create_analysis_ready_datasets(df):
    """Create specialized datasets for different analyses"""
    print("\n" + "="*80)
    print("CREATING ANALYSIS-READY DATASETS")
    print("="*80)
    
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    # 1. Dataset for SOH modeling (focus on key features)
    soh_features = [
        'vehicle_id', 'chemistry', 'vehicle_type',
        'vhc_speed', 'hv_voltage', 'hv_current', 'bcell_soc',
        'bcell_maxVoltage', 'bcell_minVoltage', 'cell_imbalance',
        'bcell_maxTemp', 'bcell_minTemp', 'temp_gradient',
        'vhc_totalMile', 'soh_capacity',
        'power_kw', 'soc_category', 'driving_intensity'
    ]
    
    available_features = [col for col in soh_features if col in df.columns]
    df_soh = df[available_features].copy()
    df_soh.to_parquet(data_path / "soh_modeling_dataset.parquet", index=False)
    print(f"1. SOH Modeling Dataset: {df_soh.shape} - Saved")
    
    # 2. Dataset for per-vehicle analysis
    df_by_vehicle = {}
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_df = df[df['vehicle_id'] == vehicle_id].copy()
        df_by_vehicle[vehicle_id] = vehicle_df
        vehicle_df.to_parquet(data_path / f"vehicle_{vehicle_id}_analysis.parquet", index=False)
    print(f"2. Per-vehicle datasets: {len(df_by_vehicle)} files saved")
    
    # 3. Dataset for charge/discharge analysis
    if 'charging_signal' in df.columns:
        # Charging events
        charging_df = df[df['charging_signal'] == 1].copy()
        if len(charging_df) > 0:
            charging_df.to_parquet(data_path / "charging_analysis_dataset.parquet", index=False)
            print(f"3. Charging Analysis Dataset: {charging_df.shape} - Saved")
        
        # Driving events
        driving_df = df[df['charging_signal'] == 3].copy()
        if len(driving_df) > 0:
            driving_df.to_parquet(data_path / "driving_analysis_dataset.parquet", index=False)
            print(f"4. Driving Analysis Dataset: {driving_df.shape} - Saved")
    
    # 4. Dataset by chemistry
    if 'chemistry' in df.columns:
        for chem in df['chemistry'].unique():
            chem_df = df[df['chemistry'] == chem].copy()
            chem_df.to_parquet(data_path / f"{chem}_chemistry_dataset.parquet", index=False)
        print(f"5. Chemistry datasets: {df['chemistry'].nunique()} files saved")
    
    # 5. Create sample for quick testing (10%)
    df_sample = df.sample(frac=0.1, random_state=42)
    df_sample.to_parquet(data_path / "sample_dataset_10percent.parquet", index=False)
    print(f"6. Sample Dataset (10%): {df_sample.shape} - For quick testing")
    
    # 7. Create time-series focused dataset
    time_series_features = ['vehicle_id', 'time', 'session_change', 'relative_time', 'time_diff',
                           'vhc_speed', 'bcell_soc', 'power_kw', 'bcell_maxTemp']
    available_ts_features = [col for col in time_series_features if col in df.columns]
    if available_ts_features:
        df_ts = df[available_ts_features].copy()
        df_ts.to_parquet(data_path / "time_series_dataset.parquet", index=False)
        print(f"7. Time Series Dataset: {df_ts.shape} - Saved")
    
    return {
        'soh_dataset': df_soh,
        'by_vehicle': df_by_vehicle,
        'sample': df_sample
    }

def main():
    """Main cleaning pipeline"""
    print("STARTING COMPREHENSIVE DATA CLEANING PIPELINE")
    print("="*80)
    
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    # Step 1: Load original data for comparison
    print("Loading original dataset...")
    df_original = pd.read_parquet(data_path / "all_vehicles_combined.parquet")
    print(f"Original shape: {df_original.shape}")
    
    # Step 2: Clean data
    df_cleaned = load_and_clean_data()
    
    # Step 3: Diagnose missing value issues
    diagnose_missing_value_issues(df_original, df_cleaned)
    
    # Step 4: Verify cleaning
    verification_log = verify_cleaning(df_cleaned)
    
    # Step 5: Create analysis-ready datasets
    analysis_datasets = create_analysis_ready_datasets(df_cleaned)
    
    print("\n" + "="*80)
    print("CLEANING PIPELINE COMPLETE!")
    print("="*80)
    
    # Summary
    print(f"\nGenerated Files in {data_path}:")
    print("1. all_vehicles_cleaned.parquet - Fully cleaned & balanced dataset")
    print("2. soh_modeling_dataset.parquet - Optimized for SOH prediction")
    print("3. vehicle_*_analysis.parquet - Individual vehicle datasets")
    print("4. charging_analysis_dataset.parquet - Charging events only")
    print("5. driving_analysis_dataset.parquet - Driving events only")
    print("6. *_chemistry_dataset.parquet - Chemistry-based datasets")
    print("7. sample_dataset_10percent.parquet - 10% sample for quick testing")
    print("8. time_series_dataset.parquet - Time-series focused data")
    
    print("\nLog Files:")
    print("1. data_cleaning_log.txt - Detailed cleaning steps")
    print("2. data_cleaning_verification.txt - Cleaning verification")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Original dataset: {df_original.shape}")
    print(f"Cleaned dataset: {df_cleaned.shape}")
    print(f"Rows removed: {df_original.shape[0] - df_cleaned.shape[0]:,} ({((df_original.shape[0] - df_cleaned.shape[0])/df_original.shape[0]*100):.1f}%)")
    print(f"Columns added: {df_cleaned.shape[1] - df_original.shape[1]}")
    
    missing_original = df_original.isnull().sum().sum()
    missing_cleaned = df_cleaned.isnull().sum().sum()
    missing_reduction = ((missing_original - missing_cleaned) / missing_original * 100) if missing_original > 0 else 0
    
    print(f"Missing values: {missing_original:,} → {missing_cleaned:,} ({missing_reduction:+.1f}% change)")
    
    return df_cleaned, analysis_datasets

if __name__ == "__main__":
    cleaned_data, analysis_sets = main()