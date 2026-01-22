# file: 02_charging_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs(B)")
OUTPUT_PATH = Path(r'C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\improved_approach\dataset_analysis_before\results_before')
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

def load_and_preprocess(vehicle_num):
    """Load and preprocess vehicle data"""
    file_path = DATA_PATH / f"vehicle#{vehicle_num}.xlsx"
    df = pd.read_excel(file_path)
    
    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Handle time column
    time_col = [col for col in df.columns if 'time' in col][0]
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    
    return df

def detect_charging_events(df, vehicle_num):
    """Detect and analyze charging events"""
    events = []
    
    # Identify charging signal column
    charging_col = [col for col in df.columns if 'charging' in col][0]
    
    # Identify current column
    current_col = [col for col in df.columns if 'current' in col and 'bcell' not in col][0]
    
    # Identify SOC column
    soc_col = [col for col in df.columns if 'soc' in col][0]
    
    # Convert charging signal to consistent format
    # Based on README: 3 = driving, 1 = charging
    # But Vehicle#7 uses 0 = charging
    if vehicle_num == 7:
        is_charging = df[charging_col] == 0
    else:
        is_charging = df[charging_col] == 1
    
    # Also use current sign as backup (negative often means charging)
    if current_col in df.columns:
        is_charging = is_charging | (df[current_col] < -1)  # More than 1A charging
    
    # Find charging segments
    charging_segments = (is_charging != is_charging.shift()).cumsum()
    
    for seg_id, segment in df.groupby(charging_segments):
        if is_charging.iloc[segment.index[0]]:  # This is a charging segment
            if len(segment) > 10:  # Minimum length filter
                event = {
                    'vehicle': vehicle_num,
                    'segment_id': seg_id,
                    'start_time': segment.iloc[0]['time'],
                    'end_time': segment.iloc[-1]['time'],
                    'duration_hours': (segment.iloc[-1]['time'] - segment.iloc[0]['time']) / 3600,
                    'start_soc': segment.iloc[0][soc_col],
                    'end_soc': segment.iloc[-1][soc_col],
                    'delta_soc': segment.iloc[-1][soc_col] - segment.iloc[0][soc_col],
                    'avg_current': segment[current_col].mean(),
                    'min_current': segment[current_col].min(),
                    'max_current': segment[current_col].max(),
                    'num_samples': len(segment)
                }
                
                # Calculate integrated current (Ah)
                if 'time' in segment.columns:
                    # Convert to timedelta for integration
                    time_diff = segment['time'].diff().fillna(0) / 3600  # Convert to hours
                    integrated_current = (segment[current_col].abs() * time_diff).sum()
                    event['integrated_current_ah'] = integrated_current
                    
                    # Estimate capacity if ΔSOC > 5%
                    if event['delta_soc'] > 5:
                        event['estimated_capacity_ah'] = integrated_current / (event['delta_soc'] / 100)
                
                events.append(event)
    
    return pd.DataFrame(events)

def analyze_charging_patterns(charging_events_df):
    """Analyze charging patterns across vehicles"""
    analysis = {}
    
    # Group by vehicle
    for vehicle_num, vehicle_events in charging_events_df.groupby('vehicle'):
        if len(vehicle_events) > 0:
            analysis[vehicle_num] = {
                'total_events': len(vehicle_events),
                'avg_duration_hours': vehicle_events['duration_hours'].mean(),
                'avg_delta_soc': vehicle_events['delta_soc'].mean(),
                'total_charge_ah': vehicle_events['integrated_current_ah'].sum(),
                'median_capacity_ah': vehicle_events['estimated_capacity_ah'].median() if 'estimated_capacity_ah' in vehicle_events.columns else None,
                'capacity_std': vehicle_events['estimated_capacity_ah'].std() if 'estimated_capacity_ah' in vehicle_events.columns else None
            }
    
    return pd.DataFrame(analysis).T

def plot_charging_analysis(charging_events_df, vehicle_analysis):
    """Create visualization of charging analysis"""
    
    # 1. Charging events per vehicle
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Number of charging events
    ax1 = axes[0, 0]
    vehicle_analysis['total_events'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Number of Charging Events per Vehicle')
    ax1.set_xlabel('Vehicle')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Average charging duration
    ax2 = axes[0, 1]
    vehicle_analysis['avg_duration_hours'].plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Average Charging Duration (Hours)')
    ax2.set_xlabel('Vehicle')
    ax2.set_ylabel('Hours')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Average SOC increase per charge
    ax3 = axes[0, 2]
    vehicle_analysis['avg_delta_soc'].plot(kind='bar', ax=ax3, color='salmon')
    ax3.set_title('Average SOC Increase per Charge')
    ax3.set_xlabel('Vehicle')
    ax3.set_ylabel('ΔSOC (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Capacity estimates distribution (if available)
    if 'estimated_capacity_ah' in charging_events_df.columns:
        ax4 = axes[1, 0]
        capacity_data = charging_events_df.dropna(subset=['estimated_capacity_ah'])
        
        # Box plot by vehicle
        box_data = []
        labels = []
        for vehicle_num in sorted(capacity_data['vehicle'].unique()):
            vehicle_capacities = capacity_data[capacity_data['vehicle'] == vehicle_num]['estimated_capacity_ah']
            if len(vehicle_capacities) > 0:
                box_data.append(vehicle_capacities.values)
                labels.append(f"V{vehicle_num}")
        
        ax4.boxplot(box_data, labels=labels)
        ax4.set_title('Capacity Estimates Distribution')
        ax4.set_xlabel('Vehicle')
        ax4.set_ylabel('Capacity (Ah)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add reference lines for initial capacities from README
        initial_capacities = {
            1: 150, 2: 150, 3: 160, 4: 160, 5: 160, 6: 160,
            7: 120, 8: 645, 9: 505, 10: 505
        }
        
        for idx, vehicle_num in enumerate(sorted(capacity_data['vehicle'].unique()), 1):
            if vehicle_num in initial_capacities:
                ax4.axhline(y=initial_capacities[vehicle_num], xmin=(idx-1)/len(labels), 
                           xmax=idx/len(labels), color='red', linestyle='--', alpha=0.5)
    
    # Plot 5: SOC vs Capacity scatter
    if 'estimated_capacity_ah' in charging_events_df.columns:
        ax5 = axes[1, 1]
        scatter_data = charging_events_df.dropna(subset=['estimated_capacity_ah', 'delta_soc'])
        
        scatter = ax5.scatter(scatter_data['delta_soc'], scatter_data['estimated_capacity_ah'],
                             c=scatter_data['vehicle'], cmap='tab10', alpha=0.6)
        ax5.set_title('Capacity Estimate vs ΔSOC')
        ax5.set_xlabel('ΔSOC (%)')
        ax5.set_ylabel('Estimated Capacity (Ah)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Vehicle')
    
    # Plot 6: Charging current distribution
    ax6 = axes[1, 2]
    for vehicle_num in sorted(charging_events_df['vehicle'].unique())[:5]:  # First 5 vehicles
        vehicle_data = charging_events_df[charging_events_df['vehicle'] == vehicle_num]
        ax6.hist(vehicle_data['avg_current'].abs(), alpha=0.5, label=f'V{vehicle_num}', bins=20)
    
    ax6.set_title('Charging Current Distribution')
    ax6.set_xlabel('Average Current (A)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'charging_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual vehicle charging timeline
    for vehicle_num in charging_events_df['vehicle'].unique():
        vehicle_events = charging_events_df[charging_events_df['vehicle'] == vehicle_num]
        
        if len(vehicle_events) > 5:  # Only plot if enough events
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert time to datetime-like progression
            vehicle_events = vehicle_events.sort_values('start_time')
            vehicle_events['event_num'] = range(len(vehicle_events))
            
            # Plot capacity estimates over time
            if 'estimated_capacity_ah' in vehicle_events.columns:
                valid_capacity = vehicle_events.dropna(subset=['estimated_capacity_ah'])
                if len(valid_capacity) > 0:
                    ax.scatter(valid_capacity['event_num'], valid_capacity['estimated_capacity_ah'],
                              color='blue', alpha=0.6, label='Capacity Estimate')
                    
                    # Add trend line
                    z = np.polyfit(valid_capacity['event_num'], valid_capacity['estimated_capacity_ah'], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_capacity['event_num'], p(valid_capacity['event_num']),
                           "r--", alpha=0.8, label='Trend')
            
            ax.set_title(f'Vehicle #{vehicle_num} - Capacity Estimates Over Charging Events')
            ax.set_xlabel('Charging Event Number')
            ax.set_ylabel('Estimated Capacity (Ah)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH / f'vehicle{vehicle_num}_capacity_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function for charging analysis"""
    print("=" * 60)
    print("CHARGING EVENT ANALYSIS")
    print("=" * 60)
    
    all_charging_events = []
    
    # Analyze each vehicle
    for vehicle_num in range(1, 11):
        print(f"\nAnalyzing charging events for Vehicle #{vehicle_num}...")
        
        try:
            df = load_and_preprocess(vehicle_num)
            charging_events = detect_charging_events(df, vehicle_num)
            
            if len(charging_events) > 0:
                all_charging_events.append(charging_events)
                print(f"  Found {len(charging_events)} charging events")
                print(f"  Average ΔSOC: {charging_events['delta_soc'].mean():.1f}%")
                print(f"  Average duration: {charging_events['duration_hours'].mean():.2f} hours")
                
                if 'estimated_capacity_ah' in charging_events.columns:
                    valid_capacity = charging_events['estimated_capacity_ah'].dropna()
                    if len(valid_capacity) > 0:
                        print(f"  Capacity estimates: {len(valid_capacity)} events")
                        print(f"  Median capacity: {valid_capacity.median():.1f} Ah")
                        print(f"  Capacity std: {valid_capacity.std():.1f} Ah")
            else:
                print(f"  No charging events detected")
                
        except Exception as e:
            print(f"  Error analyzing Vehicle #{vehicle_num}: {e}")
    
    # Combine all events
    if all_charging_events:
        charging_events_df = pd.concat(all_charging_events, ignore_index=True)
        
        # Save charging events
        charging_events_df.to_csv(OUTPUT_PATH / 'all_charging_events.csv', index=False)
        print(f"\nSaved charging events to: {OUTPUT_PATH / 'all_charging_events.csv'}")
        
        # Analyze patterns
        vehicle_analysis = analyze_charging_patterns(charging_events_df)
        vehicle_analysis.to_csv(OUTPUT_PATH / 'vehicle_charging_analysis.csv')
        print(f"Saved vehicle analysis to: {OUTPUT_PATH / 'vehicle_charging_analysis.csv'}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        plot_charging_analysis(charging_events_df, vehicle_analysis)
        print(f"Visualizations saved to: {OUTPUT_PATH}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("CHARGING ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nTotal charging events detected: {len(charging_events_df)}")
        
        if 'estimated_capacity_ah' in charging_events_df.columns:
            capacity_events = charging_events_df.dropna(subset=['estimated_capacity_ah'])
            print(f"Events with capacity estimates: {len(capacity_events)}")
            
            # Calculate potential SOH if we have initial capacities
            initial_capacities = {
                1: 150, 2: 150, 3: 160, 4: 160, 5: 160, 6: 160,
                7: 120, 8: 645, 9: 505, 10: 505
            }
            
            print(f"\nEstimated SOH ranges per vehicle (if capacity > 0):")
            for vehicle_num in sorted(capacity_events['vehicle'].unique()):
                if vehicle_num in initial_capacities:
                    vehicle_capacities = capacity_events[capacity_events['vehicle'] == vehicle_num]['estimated_capacity_ah']
                    if len(vehicle_capacities) > 0:
                        soh_values = (vehicle_capacities / initial_capacities[vehicle_num]) * 100
                        print(f"  Vehicle #{vehicle_num}: {soh_values.min():.1f}% - {soh_values.max():.1f}% "
                              f"(median: {soh_values.median():.1f}%)")
    
    else:
        print("No charging events were detected in any vehicle.")

if __name__ == "__main__":
    main()