"""
Time Series Analysis
Analyze temporal patterns in battery data (despite broken timestamps)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

def load_cleaned_data():
    """Load the cleaned dataset"""
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    df = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet")
    return df

def analyze_time_series_structure(df):
    """Analyze the structure of time series data"""
    print("="*80)
    print("TIME SERIES STRUCTURE ANALYSIS")
    print("="*80)
    
    if 'time' not in df.columns:
        print("❌ Time column not found!")
        return None
    
    # Analyze time structure
    time_analysis = {}
    
    # Basic statistics
    time_analysis['min_time'] = df['time'].min()
    time_analysis['max_time'] = df['time'].max()
    time_analysis['time_range'] = time_analysis['max_time'] - time_analysis['min_time']
    time_analysis['unique_timestamps'] = df['time'].nunique()
    time_analysis['duplicate_timestamps'] = len(df) - time_analysis['unique_timestamps']
    
    print(f"\nTime Range: {time_analysis['min_time']} to {time_analysis['max_time']}")
    print(f"Time Span: {time_analysis['time_range']:,} units")
    print(f"Unique Timestamps: {time_analysis['unique_timestamps']:,}")
    print(f"Duplicate Timestamps: {time_analysis['duplicate_timestamps']:,}")
    
    # Check if we have relative_time from cleaning
    if 'relative_time' in df.columns:
        print(f"\n✅ Relative time column found from cleaning")
        print(f"   Max relative time: {df['relative_time'].max():,}")
        print(f"   Unique sessions: {df['session_change'].nunique():,}" if 'session_change' in df.columns else "")
    
    # Analyze sampling intervals
    if 'time_diff' in df.columns:
        time_diffs = df['time_diff'].dropna()
        time_analysis['mean_interval'] = time_diffs.mean()
        time_analysis['median_interval'] = time_diffs.median()
        time_analysis['std_interval'] = time_diffs.std()
        time_analysis['min_interval'] = time_diffs.min()
        time_analysis['max_interval'] = time_diffs.max()
        
        print(f"\nTime Interval Statistics:")
        print(f"  Mean interval: {time_analysis['mean_interval']:.2f}")
        print(f"  Median interval: {time_analysis['median_interval']:.2f}")
        print(f"  Std interval: {time_analysis['std_interval']:.2f}")
        print(f"  Min interval: {time_analysis['min_interval']:.2f}")
        print(f"  Max interval: {time_analysis['max_interval']:.2f}")
        
        # Check for regular sampling
        unique_intervals = time_diffs.round(2).unique()
        print(f"  Unique intervals: {len(unique_intervals)}")
        if len(unique_intervals) <= 10:
            print(f"  Most common intervals: {pd.Series(time_diffs.round(2)).value_counts().head(5).to_dict()}")
    
    # Check for gaps in data
    if 'time' in df.columns and 'vehicle_id' in df.columns:
        gaps_by_vehicle = {}
        for vehicle_id in df['vehicle_id'].unique():
            vehicle_times = df[df['vehicle_id'] == vehicle_id]['time'].sort_values()
            if len(vehicle_times) > 1:
                gaps = vehicle_times.diff().dropna()
                large_gaps = (gaps > gaps.median() * 10).sum()
                gaps_by_vehicle[vehicle_id] = {
                    'total_gaps': len(gaps),
                    'large_gaps': int(large_gaps),
                    'max_gap': gaps.max()
                }
        
        print(f"\nTime Gaps by Vehicle:")
        for vid, gaps in sorted(gaps_by_vehicle.items()):
            print(f"  Vehicle {vid}: {gaps['large_gaps']} large gaps, max gap: {gaps['max_gap']:.0f}")
    
    return time_analysis

def visualize_time_patterns(df, vehicle_id=1):
    """Visualize time patterns for a specific vehicle"""
    print(f"\n" + "="*80)
    print(f"TIME PATTERNS VISUALIZATION - Vehicle {vehicle_id}")
    print("="*80)
    
    if vehicle_id not in df['vehicle_id'].unique():
        print(f"❌ Vehicle {vehicle_id} not found in dataset!")
        return None
    
    vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
    if len(vehicle_data) < 100:
        print(f"⚠️ Vehicle {vehicle_id} has only {len(vehicle_data)} data points")
        return None
    
    # Sort by time
    vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Time Series Analysis - Vehicle {vehicle_id}', fontsize=16, fontweight='bold')
    
    # 1. Time gaps distribution
    ax1 = axes[0, 0]
    if 'time_diff' in vehicle_data.columns:
        time_diffs = vehicle_data['time_diff'].dropna()
        ax1.hist(time_diffs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Time Difference', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Time Intervals', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {time_diffs.mean():.1f}\nMedian: {time_diffs.median():.1f}\nStd: {time_diffs.std():.1f}"
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. SOC over time
    ax2 = axes[0, 1]
    if 'bcell_soc' in vehicle_data.columns and 'relative_time' in vehicle_data.columns:
        # Sample for plotting if too many points
        plot_data = vehicle_data.sample(min(5000, len(vehicle_data))) if len(vehicle_data) > 5000 else vehicle_data
        
        ax2.scatter(plot_data['relative_time'], plot_data['bcell_soc'], 
                   alpha=0.5, s=1, color='green')
        ax2.set_xlabel('Relative Time', fontsize=12)
        ax2.set_ylabel('SOC (%)', fontsize=12)
        ax2.set_title('SOC vs Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add moving average
        if len(vehicle_data) > 100:
            window = min(100, len(vehicle_data) // 10)
            soc_ma = vehicle_data['bcell_soc'].rolling(window=window, center=True).mean()
            ax2.plot(vehicle_data['relative_time'], soc_ma, 'r-', linewidth=2, label=f'{window}-point MA')
            ax2.legend()
    
    # 3. Current over time
    ax3 = axes[1, 0]
    if 'hv_current' in vehicle_data.columns and 'relative_time' in vehicle_data.columns:
        plot_data = vehicle_data.sample(min(5000, len(vehicle_data))) if len(vehicle_data) > 5000 else vehicle_data
        
        # Color by charging/discharging
        colors = np.where(plot_data['hv_current'] < 0, 'red', 'blue')
        
        ax3.scatter(plot_data['relative_time'], plot_data['hv_current'], 
                   alpha=0.3, s=1, c=colors)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Relative Time', fontsize=12)
        ax3.set_ylabel('Current (A)', fontsize=12)
        ax3.set_title('Current vs Time (Red=Charging, Blue=Discharging)', fontsize=14)
        ax3.grid(True, alpha=0.3)
    
    # 4. Voltage over time
    ax4 = axes[1, 1]
    if 'hv_voltage' in vehicle_data.columns and 'relative_time' in vehicle_data.columns:
        plot_data = vehicle_data.sample(min(5000, len(vehicle_data))) if len(vehicle_data) > 5000 else vehicle_data
        
        ax4.scatter(plot_data['relative_time'], plot_data['hv_voltage'], 
                   alpha=0.5, s=1, color='purple')
        ax4.set_xlabel('Relative Time', fontsize=12)
        ax4.set_ylabel('Voltage (V)', fontsize=12)
        ax4.set_title('Voltage vs Time', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    # 5. Temperature over time
    ax5 = axes[2, 0]
    if 'bcell_maxTemp' in vehicle_data.columns and 'relative_time' in vehicle_data.columns:
        plot_data = vehicle_data.sample(min(5000, len(vehicle_data))) if len(vehicle_data) > 5000 else vehicle_data
        
        ax5.scatter(plot_data['relative_time'], plot_data['bcell_maxTemp'], 
                   alpha=0.5, s=1, color='orange', label='Max Temp')
        
        if 'bcell_minTemp' in vehicle_data.columns:
            ax5.scatter(plot_data['relative_time'], plot_data['bcell_minTemp'], 
                       alpha=0.5, s=1, color='blue', label='Min Temp')
        
        ax5.set_xlabel('Relative Time', fontsize=12)
        ax5.set_ylabel('Temperature (°C)', fontsize=12)
        ax5.set_title('Temperature vs Time', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. SOH over time
    ax6 = axes[2, 1]
    if 'soh_capacity' in vehicle_data.columns and 'relative_time' in vehicle_data.columns:
        plot_data = vehicle_data.sample(min(5000, len(vehicle_data))) if len(vehicle_data) > 5000 else vehicle_data
        
        ax6.scatter(plot_data['relative_time'], plot_data['soh_capacity'], 
                   alpha=0.5, s=1, color='brown')
        ax6.set_xlabel('Relative Time', fontsize=12)
        ax6.set_ylabel('SOH (%)', fontsize=12)
        ax6.set_title('SOH vs Time', fontsize=14)
        ax6.grid(True, alpha=0.3)
        
        # Add trend line if enough data
        if len(vehicle_data) > 100:
            z = np.polyfit(vehicle_data['relative_time'], vehicle_data['soh_capacity'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(vehicle_data['relative_time'].min(), 
                                 vehicle_data['relative_time'].max(), 100)
            ax6.plot(x_range, p(x_range), 'r--', linewidth=2, 
                    label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')
            ax6.legend()
    
    plt.tight_layout()
    plt.savefig(f'time_patterns_vehicle_{vehicle_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vehicle_data

def analyze_driving_sessions(df):
    """Analyze driving sessions patterns"""
    print("\n" + "="*80)
    print("DRIVING SESSIONS ANALYSIS")
    print("="*80)
    
    if 'session_change' not in df.columns:
        print("❌ Session information not available!")
        return None
    
    # Analyze sessions
    sessions = df['session_change'].unique()
    print(f"Total Sessions: {len(sessions):,}")
    
    # Calculate session statistics
    session_stats = []
    for session_id in sessions[:100]:  # Limit to first 100 for performance
        session_data = df[df['session_change'] == session_id]
        
        if len(session_data) > 1:
            stats = {
                'session_id': session_id,
                'vehicle_id': session_data['vehicle_id'].iloc[0],
                'duration': session_data['relative_time'].max() - session_data['relative_time'].min(),
                'data_points': len(session_data),
                'avg_speed': session_data['vhc_speed'].mean(),
                'max_speed': session_data['vhc_speed'].max(),
                'soc_start': session_data['bcell_soc'].iloc[0] if 'bcell_soc' in session_data.columns else None,
                'soc_end': session_data['bcell_soc'].iloc[-1] if 'bcell_soc' in session_data.columns else None,
                'is_driving': (session_data['vhc_speed'].mean() > 1)
            }
            
            if stats['soc_start'] is not None and stats['soc_end'] is not None:
                stats['soc_change'] = stats['soc_end'] - stats['soc_start']
            
            session_stats.append(stats)
    
    session_df = pd.DataFrame(session_stats)
    
    if not session_df.empty:
        print(f"\nSession Statistics (first {len(session_df)} sessions):")
        print(f"  Average duration: {session_df['duration'].mean():.1f} time units")
        print(f"  Average data points: {session_df['data_points'].mean():.1f}")
        print(f"  Driving sessions: {(session_df['is_driving']).sum():,}")
        print(f"  Stationary sessions: {(~session_df['is_driving']).sum():,}")
        
        if 'soc_change' in session_df.columns:
            print(f"  Average SOC change: {session_df['soc_change'].mean():.1f}%")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Session duration distribution
        ax1 = axes[0, 0]
        ax1.hist(session_df['duration'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Session Duration', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Session Durations', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. Data points per session
        ax2 = axes[0, 1]
        ax2.hist(session_df['data_points'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Data Points per Session', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Session Sizes', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Driving vs Stationary
        ax3 = axes[1, 0]
        driving_counts = session_df['is_driving'].value_counts()
        ax3.bar(['Stationary', 'Driving'], driving_counts.values, color=['skyblue', 'lightgreen'], edgecolor='black')
        ax3.set_ylabel('Number of Sessions', fontsize=12)
        ax3.set_title('Driving vs Stationary Sessions', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, count in enumerate(driving_counts.values):
            ax3.text(i, count + 5, f'{count}\n({count/len(session_df)*100:.1f}%)', 
                    ha='center', fontsize=10)
        
        # 4. SOC change distribution
        ax4 = axes[1, 1]
        if 'soc_change' in session_df.columns:
            ax4.hist(session_df['soc_change'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='gold')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_xlabel('SOC Change (%)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Distribution of SOC Changes per Session', fontsize=14)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('driving_sessions_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return session_df

def analyze_cyclic_patterns(df, vehicle_id=1):
    """Analyze cyclic patterns in battery data"""
    print(f"\n" + "="*80)
    print(f"CYCLIC PATTERNS ANALYSIS - Vehicle {vehicle_id}")
    print("="*80)
    
    if vehicle_id not in df['vehicle_id'].unique():
        print(f"❌ Vehicle {vehicle_id} not found!")
        return None
    
    vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
    if len(vehicle_data) < 1000:
        print(f"⚠️ Insufficient data for cyclic analysis")
        return None
    
    # Sort by relative time
    if 'relative_time' in vehicle_data.columns:
        vehicle_data = vehicle_data.sort_values('relative_time')
        
        # Resample to regular intervals for FFT
        resample_interval = 10  # 10 time units
        max_time = vehicle_data['relative_time'].max()
        regular_times = np.arange(0, max_time, resample_interval)
        
        # Interpolate key signals
        signals = {}
        for signal_name in ['bcell_soc', 'hv_current', 'hv_voltage', 'bcell_maxTemp']:
            if signal_name in vehicle_data.columns:
                # Simple interpolation
                from scipy import interpolate
                valid_data = vehicle_data[['relative_time', signal_name]].dropna()
                if len(valid_data) > 10:
                    f = interpolate.interp1d(valid_data['relative_time'], valid_data[signal_name], 
                                           bounds_error=False, fill_value='extrapolate')
                    signals[signal_name] = f(regular_times)
        
        # Perform FFT analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Cyclic Pattern Analysis - Vehicle {vehicle_id}', fontsize=16, fontweight='bold')
        
        for idx, (signal_name, signal_data) in enumerate(list(signals.items())[:4]):
            ax = axes[idx // 2, idx % 2]
            
            # Remove NaN values
            signal_data_clean = signal_data[~np.isnan(signal_data)]
            if len(signal_data_clean) < 10:
                continue
            
            # Compute FFT
            N = len(signal_data_clean)
            T = resample_interval
            yf = np.fft.fft(signal_data_clean - np.mean(signal_data_clean))
            xf = np.fft.fftfreq(N, T)[:N//2]
            
            # Plot frequency spectrum
            ax.plot(xf[1:], 2.0/N * np.abs(yf[0:N//2])[1:], 'b-', linewidth=1)
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title(f'Frequency Spectrum: {signal_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Find dominant frequency
            if len(xf) > 1:
                dominant_idx = np.argmax(2.0/N * np.abs(yf[0:N//2])[1:])
                dominant_freq = xf[1:][dominant_idx]
                dominant_period = 1/dominant_freq if dominant_freq > 0 else np.inf
                
                ax.axvline(x=dominant_freq, color='red', linestyle='--', alpha=0.7,
                          label=f'Dominant: {dominant_freq:.4f} Hz\nPeriod: {dominant_period:.1f} units')
                ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'cyclic_patterns_vehicle_{vehicle_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return signals
    
    return None

def analyze_seasonal_patterns(df):
    """Analyze seasonal/periodic patterns across vehicles"""
    print("\n" + "="*80)
    print("SEASONAL PATTERNS ANALYSIS")
    print("="*80)
    
    if 'relative_time' not in df.columns or 'vehicle_id' not in df.columns:
        print("❌ Required columns not found!")
        return None
    
    # Analyze patterns across multiple vehicles
    vehicles = df['vehicle_id'].unique()[:5]  # First 5 vehicles
    
    fig, axes = plt.subplots(len(vehicles), 3, figsize=(16, 4*len(vehicles)))
    if len(vehicles) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, vehicle_id in enumerate(vehicles):
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        
        if len(vehicle_data) < 100:
            continue
        
        # Sort by relative time
        vehicle_data = vehicle_data.sort_values('relative_time')
        
        # 1. SOC patterns
        ax1 = axes[idx, 0]
        ax1.scatter(vehicle_data['relative_time'], vehicle_data['bcell_soc'], 
                   alpha=0.3, s=1, color='green')
        ax1.set_xlabel('Relative Time', fontsize=10)
        ax1.set_ylabel('SOC (%)', fontsize=10)
        ax1.set_title(f'Vehicle {vehicle_id}: SOC Pattern', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Current patterns
        ax2 = axes[idx, 1]
        ax2.scatter(vehicle_data['relative_time'], vehicle_data['hv_current'], 
                   alpha=0.3, s=1, color='blue')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Relative Time', fontsize=10)
        ax2.set_ylabel('Current (A)', fontsize=10)
        ax2.set_title(f'Vehicle {vehicle_id}: Current Pattern', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Voltage patterns
        ax3 = axes[idx, 2]
        ax3.scatter(vehicle_data['relative_time'], vehicle_data['hv_voltage'], 
                   alpha=0.3, s=1, color='purple')
        ax3.set_xlabel('Relative Time', fontsize=10)
        ax3.set_ylabel('Voltage (V)', fontsize=10)
        ax3.set_title(f'Vehicle {vehicle_id}: Voltage Pattern', fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seasonal_patterns_across_vehicles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vehicles

def main():
    """Main time series analysis function"""
    print("STARTING TIME SERIES ANALYSIS")
    print("="*80)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    # Perform analyses
    time_structure = analyze_time_series_structure(df)
    
    # Analyze specific vehicles
    for vehicle_id in [1, 7]:  # Analyze Vehicle 1 (typical) and 7 (degraded)
        vehicle_patterns = visualize_time_patterns(df, vehicle_id)
    
    # Analyze driving sessions
    session_analysis = analyze_driving_sessions(df)
    
    # Analyze cyclic patterns for Vehicle 1
    cyclic_patterns = analyze_cyclic_patterns(df, vehicle_id=1)
    
    # Analyze seasonal patterns
    seasonal_patterns = analyze_seasonal_patterns(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n📈 Generated Visualizations:")
    print("1. time_patterns_vehicle_1.png - Time patterns for Vehicle 1")
    print("2. time_patterns_vehicle_7.png - Time patterns for Vehicle 7")
    print("3. driving_sessions_analysis.png - Session statistics")
    print("4. cyclic_patterns_vehicle_1.png - Frequency analysis")
    print("5. seasonal_patterns_across_vehicles.png - Multi-vehicle patterns")
    
    print("\n📊 KEY FINDINGS:")
    if time_structure:
        print(f"1. Time range: {time_structure['time_range']:,} units")
        print(f"2. Unique timestamps: {time_structure['unique_timestamps']:,}")
    
    if session_analysis is not None and not session_analysis.empty:
        driving_sessions = (session_analysis['is_driving']).sum()
        print(f"3. Driving sessions: {driving_sessions:,}")
        print(f"4. Average session duration: {session_analysis['duration'].mean():.1f} units")

if __name__ == "__main__":
    main()