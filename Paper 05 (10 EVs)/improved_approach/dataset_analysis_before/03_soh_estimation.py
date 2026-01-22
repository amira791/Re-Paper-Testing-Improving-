# file: 03_soh_estimation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(B)")
OUTPUT_PATH = Path(r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\improved_approach\dataset_analysis_before\results_before')
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# Initial capacities from README
INITIAL_CAPACITIES = {
    1: 150,  # Ah
    2: 150,
    3: 160,
    4: 160,
    5: 160,
    6: 160,
    7: 120,
    8: 645,
    9: 505,
    10: 505
}

def estimate_capacity_from_charge_event(charge_segment, vehicle_num):
    """
    Estimate battery capacity from a charging segment using the paper's method
    C_now = ∫I(t)dt / ΔSOC
    """
    # Required columns
    required_cols = ['current', 'soc', 'time']
    
    # Check if we have the right column names
    col_mapping = {}
    for req_col in required_cols:
        matching_cols = [col for col in charge_segment.columns if req_col in col.lower()]
        if matching_cols:
            col_mapping[req_col] = matching_cols[0]
        else:
            return None  # Missing required column
    
    current_col = col_mapping['current']
    soc_col = col_mapping['soc']
    time_col = col_mapping['time']
    
    # Ensure numeric types
    charge_segment = charge_segment.copy()
    charge_segment[current_col] = pd.to_numeric(charge_segment[current_col], errors='coerce')
    charge_segment[soc_col] = pd.to_numeric(charge_segment[soc_col], errors='coerce')
    charge_segment[time_col] = pd.to_numeric(charge_segment[time_col], errors='coerce')
    
    # Drop rows with NaN in critical columns
    charge_segment = charge_segment.dropna(subset=[current_col, soc_col, time_col])
    
    if len(charge_segment) < 2:
        return None
    
    # Calculate ΔSOC
    soc_start = charge_segment.iloc[0][soc_col]
    soc_end = charge_segment.iloc[-1][soc_col]
    delta_soc = soc_end - soc_start
    
    # Filter for reasonable ΔSOC values
    if delta_soc < 5:  # Minimum 5% SOC change for reliable estimate
        return None
    
    if delta_soc > 100:  # Should not exceed 100%
        return None
    
    # Calculate ∫I(t)dt (current integration)
    # Current is typically negative for charging, use absolute value
    current_values = np.abs(charge_segment[current_col].values)
    time_values = charge_segment[time_col].values
    
    # Calculate time differences in hours
    time_diffs = np.diff(time_values) / 3600.0  # Convert seconds to hours
    
    # Integrate using trapezoidal rule
    # For each interval: area = (I1 + I2)/2 * Δt
    integrated_current = 0
    for i in range(len(time_diffs)):
        avg_current = (current_values[i] + current_values[i+1]) / 2
        integrated_current += avg_current * time_diffs[i]
    
    # Additional integration for last point if needed
    if len(current_values) > len(time_diffs) + 1:
        # Use last time diff as approximation
        avg_current = (current_values[-2] + current_values[-1]) / 2
        integrated_current += avg_current * time_diffs[-1]
    
    # Estimate current capacity
    estimated_capacity = integrated_current / (delta_soc / 100.0)
    
    # Filter unreasonable capacity estimates
    initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
    if estimated_capacity < initial_capacity * 0.5:  # Less than 50% of initial
        return None
    if estimated_capacity > initial_capacity * 1.2:  # More than 120% of initial
        return None
    
    return {
        'vehicle': vehicle_num,
        'start_time': time_values[0],
        'end_time': time_values[-1],
        'duration_hours': (time_values[-1] - time_values[0]) / 3600,
        'soc_start': soc_start,
        'soc_end': soc_end,
        'delta_soc': delta_soc,
        'integrated_current_ah': integrated_current,
        'estimated_capacity_ah': estimated_capacity,
        'soh_percent': (estimated_capacity / initial_capacity) * 100,
        'num_samples': len(charge_segment)
    }

def detect_charging_segments(df, vehicle_num):
    """Detect charging segments in vehicle data"""
    charging_segments = []
    
    # Standardize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Identify relevant columns
    time_col = [col for col in df.columns if 'time' in col][0]
    current_col = [col for col in df.columns if 'current' in col and 'bcell' not in col][0]
    soc_col = [col for col in df.columns if 'soc' in col][0]
    
    # Detect charging based on multiple criteria
    # 1. Charging signal (if available)
    charging_signal_col = None
    charging_cols = [col for col in df.columns if 'charging' in col]
    if charging_cols:
        charging_signal_col = charging_cols[0]
    
    # 2. Current negative (charging) and vehicle stationary
    speed_col = None
    speed_cols = [col for col in df.columns if 'speed' in col]
    if speed_cols:
        speed_col = speed_cols[0]
    
    # Create charging mask
    charging_mask = pd.Series(False, index=df.index)
    
    # Criterion 1: Negative current (more than 1A in magnitude)
    if current_col in df.columns:
        charging_mask = charging_mask | (df[current_col] < -1)
    
    # Criterion 2: Charging signal (vehicle specific)
    if charging_signal_col:
        if vehicle_num == 7:
            # Vehicle#7 uses 0 for charging
            charging_mask = charging_mask | (df[charging_signal_col] == 0)
        else:
            # Others use 1 for charging
            charging_mask = charging_mask | (df[charging_signal_col] == 1)
    
    # Criterion 3: Vehicle stationary (if speed data available)
    if speed_col and speed_col in df.columns:
        charging_mask = charging_mask & (df[speed_col] < 1)  # Less than 1 km/h
    
    # Group consecutive charging points
    charging_groups = (charging_mask != charging_mask.shift()).cumsum()
    
    for group_id, group_data in df.groupby(charging_groups):
        if charging_mask.iloc[group_data.index[0]]:  # This is a charging group
            if len(group_data) >= 10:  # Minimum group size
                charging_segments.append(group_data)
    
    return charging_segments

def process_vehicle_soh(vehicle_num):
    """Process SOH estimation for a single vehicle"""
    print(f"  Processing Vehicle #{vehicle_num}...")
    
    try:
        # Load data
        file_path = DATA_PATH / f"vehicle#{vehicle_num}.xlsx"
        df = pd.read_excel(file_path)
        
        # Detect charging segments
        charging_segments = detect_charging_segments(df, vehicle_num)
        print(f"    Detected {len(charging_segments)} potential charging segments")
        
        # Estimate capacity for each segment
        capacity_estimates = []
        for i, segment in enumerate(charging_segments):
            estimate = estimate_capacity_from_charge_event(segment, vehicle_num)
            if estimate:
                estimate['segment_id'] = i
                capacity_estimates.append(estimate)
        
        print(f"    Obtained {len(capacity_estimates)} valid capacity estimates")
        
        if capacity_estimates:
            estimates_df = pd.DataFrame(capacity_estimates)
            
            # Add mileage if available
            mileage_cols = [col for col in df.columns if 'mile' in col.lower()]
            if mileage_cols:
                mileage_col = mileage_cols[0]
                # Get approximate mileage for each estimate
                for idx, row in estimates_df.iterrows():
                    # Find closest time in original data
                    time_diff = np.abs(df['time'] - row['start_time'])
                    closest_idx = time_diff.idxmin()
                    estimates_df.at[idx, 'mileage'] = df.iloc[closest_idx][mileage_col]
            
            # Smooth SOH estimates
            estimates_df = estimates_df.sort_values('start_time')
            
            # Apply Savitzky-Golay filter for smoothing
            if len(estimates_df) > 5:
                window_length = min(7, len(estimates_df) - 2 if len(estimates_df) % 2 == 0 else len(estimates_df) - 1)
                if window_length >= 3:
                    estimates_df['soh_smoothed'] = savgol_filter(
                        estimates_df['soh_percent'].values,
                        window_length=window_length,
                        polyorder=2
                    )
                else:
                    estimates_df['soh_smoothed'] = estimates_df['soh_percent']
            else:
                estimates_df['soh_smoothed'] = estimates_df['soh_percent']
            
            return estimates_df
        else:
            return None
            
    except Exception as e:
        print(f"    Error processing Vehicle #{vehicle_num}: {e}")
        return None

def plot_soh_results(all_estimates):
    """Plot SOH estimation results"""
    
    # 1. SOH trends for all vehicles
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: SOH vs Time/Mileage for each vehicle
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (vehicle_num, vehicle_df) in enumerate(all_estimates.items()):
        if vehicle_df is not None and len(vehicle_df) > 0:
            # Use mileage if available, otherwise use event number
            if 'mileage' in vehicle_df.columns:
                x_data = vehicle_df['mileage']
                x_label = 'Mileage (km)'
            else:
                x_data = range(len(vehicle_df))
                x_label = 'Charging Event'
            
            color = colors[idx % len(colors)]
            ax1.scatter(x_data, vehicle_df['soh_percent'], 
                       alpha=0.5, color=color, s=30, label=f'V{vehicle_num}')
            
            # Plot smoothed trend
            ax1.plot(x_data, vehicle_df['soh_smoothed'], 
                    color=color, linewidth=2, alpha=0.8)
    
    ax1.set_title('SOH Estimation Across All Vehicles')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('SOH (%)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SOH distribution by vehicle
    ax2 = axes[0, 1]
    soh_data = []
    vehicle_labels = []
    
    for vehicle_num, vehicle_df in all_estimates.items():
        if vehicle_df is not None and len(vehicle_df) > 0:
            soh_data.append(vehicle_df['soh_percent'].values)
            vehicle_labels.append(f'V{vehicle_num}')
    
    if soh_data:
        box = ax2.boxplot(soh_data, labels=vehicle_labels, patch_artist=True)
        # Add colors to boxes
        for patch, color in zip(box['boxes'], colors[:len(soh_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_title('SOH Distribution by Vehicle')
        ax2.set_xlabel('Vehicle')
        ax2.set_ylabel('SOH (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Capacity vs ΔSOC relationship
    ax3 = axes[1, 0]
    for vehicle_num, vehicle_df in all_estimates.items():
        if vehicle_df is not None and len(vehicle_df) > 0:
            color = colors[(vehicle_num-1) % len(colors)]
            ax3.scatter(vehicle_df['delta_soc'], vehicle_df['estimated_capacity_ah'],
                       alpha=0.6, color=color, s=40, label=f'V{vehicle_num}')
    
    ax3.set_title('Capacity Estimate vs ΔSOC')
    ax3.set_xlabel('ΔSOC (%)')
    ax3.set_ylabel('Estimated Capacity (Ah)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: SOH degradation rate
    ax4 = axes[1, 1]
    degradation_rates = []
    vehicle_nums = []
    
    for vehicle_num, vehicle_df in all_estimates.items():
        if vehicle_df is not None and len(vehicle_df) > 3:
            if 'mileage' in vehicle_df.columns:
                # Calculate degradation per 10,000 km
                x = vehicle_df['mileage'].values
                y = vehicle_df['soh_smoothed'].values
                
                if len(x) > 1 and (x[-1] - x[0]) > 1000:  # At least 1000 km range
                    # Linear fit
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    degradation_per_10kkm = slope * 10000  # SOH change per 10,000 km
                    degradation_rates.append(degradation_per_10kkm)
                    vehicle_nums.append(vehicle_num)
    
    if degradation_rates:
        bars = ax4.bar(range(len(degradation_rates)), degradation_rates, 
                      color=colors[:len(degradation_rates)])
        
        # Add value labels on bars
        for bar, rate in zip(bars, degradation_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        ax4.set_title('Estimated SOH Degradation per 10,000 km')
        ax4.set_xlabel('Vehicle')
        ax4.set_ylabel('ΔSOH per 10,000 km (%)')
        ax4.set_xticks(range(len(degradation_rates)))
        ax4.set_xticklabels([f'V{v}' for v in vehicle_nums])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'soh_estimation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual vehicle plots
    for vehicle_num, vehicle_df in all_estimates.items():
        if vehicle_df is not None and len(vehicle_df) > 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: SOH trend
            if 'mileage' in vehicle_df.columns:
                x_data = vehicle_df['mileage']
                x_label = 'Mileage (km)'
            else:
                x_data = vehicle_df['start_time'] / (3600 * 24)  # Convert to days
                x_label = 'Time (days from start)'
            
            ax1.scatter(x_data, vehicle_df['soh_percent'], alpha=0.6, 
                       color='blue', s=40, label='Raw estimates')
            ax1.plot(x_data, vehicle_df['soh_smoothed'], 'r-', 
                    linewidth=2, label='Smoothed trend')
            ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Initial (100%)')
            ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='EOL threshold (80%)')
            
            initial_capacity = INITIAL_CAPACITIES.get(vehicle_num, 150)
            ax1.set_title(f'Vehicle #{vehicle_num} - SOH Estimation\nInitial Capacity: {initial_capacity} Ah')
            ax1.set_xlabel(x_label)
            ax1.set_ylabel('SOH (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Capacity vs ΔSOC
            ax2.scatter(vehicle_df['delta_soc'], vehicle_df['estimated_capacity_ah'],
                       alpha=0.6, color='purple', s=50)
            
            # Add regression line
            if len(vehicle_df) > 2:
                z = np.polyfit(vehicle_df['delta_soc'], vehicle_df['estimated_capacity_ah'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(vehicle_df['delta_soc'].min(), vehicle_df['delta_soc'].max(), 100)
                ax2.plot(x_range, p(x_range), 'r--', alpha=0.8, 
                        label=f'Slope: {z[0]:.2f} Ah/%')
            
            ax2.set_title(f'Vehicle #{vehicle_num} - Capacity vs ΔSOC')
            ax2.set_xlabel('ΔSOC (%)')
            ax2.set_ylabel('Estimated Capacity (Ah)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH / f'vehicle{vehicle_num}_soh_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function for SOH estimation"""
    print("=" * 60)
    print("SOH ESTIMATION USING PAPER'S METHOD")
    print("=" * 60)
    
    all_estimates = {}
    
    # Process each vehicle
    for vehicle_num in range(1, 11):
        estimates_df = process_vehicle_soh(vehicle_num)
        all_estimates[vehicle_num] = estimates_df
        
        if estimates_df is not None:
            # Save individual vehicle results
            estimates_df.to_csv(OUTPUT_PATH / f'vehicle{vehicle_num}_soh_estimates.csv', index=False)
            print(f"    Saved results for Vehicle #{vehicle_num}")
            
            # Print summary statistics
            print(f"    SOH range: {estimates_df['soh_percent'].min():.1f}% - {estimates_df['soh_percent'].max():.1f}%")
            print(f"    Median SOH: {estimates_df['soh_percent'].median():.1f}%")
            print(f"    Std deviation: {estimates_df['soh_percent'].std():.2f}%")
    
    # Create combined results file
    combined_results = []
    for vehicle_num, estimates_df in all_estimates.items():
        if estimates_df is not None:
            combined_results.append(estimates_df)
    
    if combined_results:
        all_estimates_df = pd.concat(combined_results, ignore_index=True)
        all_estimates_df.to_csv(OUTPUT_PATH / 'all_vehicles_soh_estimates.csv', index=False)
        print(f"\nSaved all estimates to: {OUTPUT_PATH / 'all_vehicles_soh_estimates.csv'}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_soh_results(all_estimates)
    print(f"Visualizations saved to: {OUTPUT_PATH}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SOH ESTIMATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nVehicles with successful SOH estimation:")
    for vehicle_num, estimates_df in all_estimates.items():
        if estimates_df is not None and len(estimates_df) > 0:
            initial_cap = INITIAL_CAPACITIES.get(vehicle_num, 'Unknown')
            avg_soh = estimates_df['soh_percent'].mean()
            median_cap = estimates_df['estimated_capacity_ah'].median()
            
            print(f"  Vehicle #{vehicle_num}:")
            print(f"    Initial capacity: {initial_cap} Ah")
            print(f"    Estimated capacity: {median_cap:.1f} Ah")
            print(f"    Average SOH: {avg_soh:.1f}%")
            print(f"    Number of estimates: {len(estimates_df)}")
            
            if 'mileage' in estimates_df.columns:
                mileage_range = estimates_df['mileage'].max() - estimates_df['mileage'].min()
                print(f"    Mileage range in data: {mileage_range:.0f} km")
            print()
        else:
            print(f"  Vehicle #{vehicle_num}: No valid SOH estimates")

if __name__ == "__main__":
    main()