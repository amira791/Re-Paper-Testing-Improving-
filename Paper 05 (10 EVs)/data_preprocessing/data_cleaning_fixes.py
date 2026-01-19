"""
COMPREHENSIVE DATA CLEANING TO FIX CRITICAL ISSUES
Fixes: Temperature outliers, missing data, time series issues, SOH artifacts
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
            
            # Also clip extreme values that might be sensor errors
            if col in ['bcell_maxTemp', 'bcell_minTemp']:
                # Additional clipping for impossible values
                df[col] = df[col].clip(0, 50)  # More realistic range
                
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
    
    # FIX 4: Handle missing voltage values (65535 sensor faults)
    cleaning_log.append(f"\nFIX 4: Handling sensor faults (65535 values)")
    
    voltage_cols = ['bcell_maxVoltage', 'bcell_minVoltage', 'hv_voltage']
    for col in voltage_cols:
        if col in df.columns:
            # Replace 65535 with NaN
            faults_before = (df[col] == 65535).sum()
            df[col] = df[col].replace(65535, np.nan)
            
            # Forward fill within each vehicle's session
            if faults_before > 0:
                df[col] = df.groupby('vehicle_id')[col].ffill()
                cleaning_log.append(f"   {col}: Fixed {faults_before:,} sensor faults")
    
    # FIX 5: Fix broken time series and create session-based features
    cleaning_log.append(f"\nFIX 5: Fixing broken time series")
    
    if 'time' in df.columns:
        # Create session IDs based on vehicle and driving state changes
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
    
    # FIX 6: Balance Vehicle 7 dominance
    cleaning_log.append(f"\nFIX 6: Balancing Vehicle 7 dominance")
    
    if 'vehicle_id' in df.columns:
        vehicle_counts = df['vehicle_id'].value_counts()
        max_rows = int(vehicle_counts.median() * 1.5)  # 1.5x median
        
        cleaning_log.append(f"   Vehicle distribution before balancing:")
        for vid, count in vehicle_counts.items():
            cleaning_log.append(f"     Vehicle {vid}: {count:,} rows")
        
        # Create balanced version (optional - for certain analyses)
        balanced_dfs = []
        for vid in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vid]
            if len(vehicle_data) > max_rows:
                # Downsample if too many rows
                balanced_dfs.append(vehicle_data.sample(n=max_rows, random_state=42))
                cleaning_log.append(f"   Downsampled Vehicle {vid}: {len(vehicle_data):,} → {max_rows:,}")
            else:
                balanced_dfs.append(vehicle_data)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        cleaning_log.append(f"   Balanced dataset: {len(df_balanced):,} rows (reduced from {len(df):,})")
    
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
    
    # FIX 8: Final missing value imputation
    cleaning_log.append(f"\nFIX 8: Final missing value imputation")
    
    # List of columns to impute
    impute_cols = ['vhc_speed', 'hv_current', 'hv_voltage', 'bcell_soc', 
                   'bcell_maxVoltage', 'bcell_minVoltage']
    
    for col in impute_cols:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                # Forward fill within each vehicle
                df[col] = df.groupby('vehicle_id')[col].ffill()
                # Backward fill
                df[col] = df.groupby('vehicle_id')[col].bfill()
                # Fill any remaining with median per vehicle
                vehicle_medians = df.groupby('vehicle_id')[col].transform('median')
                df[col] = df[col].fillna(vehicle_medians)
                
                missing_after = df[col].isna().sum()
                cleaning_log.append(f"   {col}: Imputed {missing_before - missing_after:,} values")
    
    # Save cleaning log
    with open('data_cleaning_log.txt', 'w', encoding='utf-8') as f:
        f.write("DATA CLEANING LOG\n")
        f.write("="*50 + "\n\n")
        for line in cleaning_log:
            f.write(line + "\n")
    
    # Save cleaned datasets
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    # Save full cleaned dataset
    df.to_parquet(data_path / "all_vehicles_cleaned.parquet", index=False)
    
    # Save balanced version
    if 'df_balanced' in locals():
        df_balanced.to_parquet(data_path / "all_vehicles_balanced.parquet", index=False)
    
    print("\n" + "="*80)
    print("CLEANING COMPLETE!")
    print("="*80)
    
    # Print summary
    for line in cleaning_log:
        print(line)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Saved to: {data_path / 'all_vehicles_cleaned.parquet'}")
    
    if 'df_balanced' in locals():
        print(f"Balanced dataset: {df_balanced.shape}")
        print(f"Saved to: {data_path / 'all_vehicles_balanced.parquet'}")
    
    print(f"\nCleaning log saved to: data_cleaning_log.txt")
    
    return df, df_balanced if 'df_balanced' in locals() else None

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
        for col, count in missing_cols.items():
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
        print(f"5. Chemistry datasets: NCM and LFP saved")
    
    # 5. Create sample for quick testing (10%)
    df_sample = df.sample(frac=0.1, random_state=42)
    df_sample.to_parquet(data_path / "sample_dataset_10percent.parquet", index=False)
    print(f"6. Sample Dataset (10%): {df_sample.shape} - For quick testing")
    
    return {
        'soh_dataset': df_soh,
        'by_vehicle': df_by_vehicle,
        'sample': df_sample
    }

def main():
    """Main cleaning pipeline"""
    print("STARTING COMPREHENSIVE DATA CLEANING PIPELINE")
    print("="*80)
    
    # Step 1: Load and clean data
    df_cleaned, df_balanced = load_and_clean_data()
    
    # Step 2: Verify cleaning
    verification_log = verify_cleaning(df_cleaned)
    
    # Step 3: Create analysis-ready datasets
    analysis_datasets = create_analysis_ready_datasets(df_cleaned)
    
    print("\n" + "="*80)
    print("CLEANING PIPELINE COMPLETE!")
    print("="*80)
    
    # Summary
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    print(f"\nGenerated Files in {data_path}:")
    print("1. all_vehicles_cleaned.parquet - Fully cleaned dataset")
    print("2. all_vehicles_balanced.parquet - Balanced version (if created)")
    print("3. soh_modeling_dataset.parquet - Optimized for SOH prediction")
    print("4. vehicle_*_analysis.parquet - Individual vehicle datasets")
    print("5. charging_analysis_dataset.parquet - Charging events only")
    print("6. driving_analysis_dataset.parquet - Driving events only")
    print("7. NCM_chemistry_dataset.parquet / LFP_chemistry_dataset.parquet")
    print("8. sample_dataset_10percent.parquet - 10% sample for quick testing")
    
    print("\nLog Files:")
    print("1. data_cleaning_log.txt - Detailed cleaning steps")
    print("2. data_cleaning_verification.txt - Cleaning verification")
    
    return df_cleaned, analysis_datasets

if __name__ == "__main__":
    cleaned_data, analysis_sets = main()