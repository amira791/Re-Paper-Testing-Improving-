"""
Vehicle Comparison Analysis
Compare different vehicles across multiple dimensions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')


# =============================================================================
# Data Loading
# =============================================================================
def load_cleaned_data():
    """Load the cleaned dataset"""
    data_path = Path(
        r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs"
    )
    df = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet")
    return df


# =============================================================================
# Vehicle Statistics Comparison
# =============================================================================
def compare_vehicle_statistics(df):
    """Compare basic statistics across all vehicles"""
    print("=" * 80)
    print("VEHICLE STATISTICS COMPARISON")
    print("=" * 80)

    if 'vehicle_id' not in df.columns:
        print("❌ Vehicle ID column not found!")
        return None

    vehicles = sorted(df['vehicle_id'].unique())
    print(f"Total Vehicles: {len(vehicles)}")
    print(f"Vehicle IDs: {vehicles}")

    vehicle_stats = {}

    for vehicle_id in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle_id]

        stats_dict = {
            'Data Points': len(vehicle_data),
            'Chemistry': vehicle_data['chemistry'].iloc[0]
            if 'chemistry' in vehicle_data.columns else 'Unknown',
            'Vehicle Type': vehicle_data['vehicle_type'].iloc[0]
            if 'vehicle_type' in vehicle_data.columns else 'Unknown',
            'Avg Mileage': vehicle_data['vhc_totalMile'].mean()
            if 'vhc_totalMile' in vehicle_data.columns else np.nan,
            'Avg SOH': vehicle_data['soh_capacity'].mean()
            if 'soh_capacity' in vehicle_data.columns else np.nan,
            'Avg SOC': vehicle_data['bcell_soc'].mean()
            if 'bcell_soc' in vehicle_data.columns else np.nan,
            'Avg Speed': vehicle_data['vhc_speed'].mean()
            if 'vhc_speed' in vehicle_data.columns else np.nan,
            'Avg Voltage': vehicle_data['hv_voltage'].mean()
            if 'hv_voltage' in vehicle_data.columns else np.nan,
            'Avg Current': vehicle_data['hv_current'].mean()
            if 'hv_current' in vehicle_data.columns else np.nan,
            'Avg Max Temp': vehicle_data['bcell_maxTemp'].mean()
            if 'bcell_maxTemp' in vehicle_data.columns else np.nan,
            'Cell Imbalance': vehicle_data['cell_imbalance'].mean()
            if 'cell_imbalance' in vehicle_data.columns else np.nan
        }

        vehicle_stats[vehicle_id] = stats_dict

    stats_df = pd.DataFrame(vehicle_stats).T
    stats_df.index.name = 'Vehicle_ID'

    print("\nVehicle Statistics Summary:")
    print(stats_df.round(2).to_string())

    stats_df.round(2).to_csv("vehicle_statistics_summary.csv")
    print("\n✅ Statistics saved to 'vehicle_statistics_summary.csv'")

    return stats_df


# =============================================================================
# Visualization
# =============================================================================
def visualize_vehicle_comparison(df, stats_df):
    """Create comprehensive visual comparison of vehicles"""
    print("\n" + "=" * 80)
    print("VISUAL VEHICLE COMPARISON")
    print("=" * 80)

    vehicles = sorted(df['vehicle_id'].unique())

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("Comprehensive Vehicle Comparison",
                 fontsize=18, fontweight="bold")

    # -------------------------------------------------------------------------
    # 1. Data volume per vehicle
    ax = axes[0, 0]
    counts = df['vehicle_id'].value_counts().sort_index()
    ax.bar(range(len(vehicles)), counts.values,
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Data Volume per Vehicle")
    ax.set_xlabel("Vehicle ID")
    ax.set_ylabel("Data Points")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 2. SOH distribution
    ax = axes[0, 1]
    if 'soh_capacity' in df.columns:
        data = [df[df.vehicle_id == v]['soh_capacity'].dropna()
                for v in vehicles]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for i, b in enumerate(box['boxes']):
            b.set_facecolor(plt.cm.tab20(i))
        ax.set_title("SOH Distribution")
        ax.set_ylabel("SOH (%)")
        ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 3. Mileage distribution
    ax = axes[0, 2]
    if 'vhc_totalMile' in df.columns:
        data = [df[df.vehicle_id == v]['vhc_totalMile'].dropna()
                for v in vehicles]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for i, b in enumerate(box['boxes']):
            b.set_facecolor(plt.cm.tab20(i))
        ax.set_title("Mileage Distribution")
        ax.set_ylabel("Mileage (km)")
        ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 4. Average speed
    ax = axes[0, 3]
    ax.bar(range(len(vehicles)), stats_df['Avg Speed'],
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Average Speed")
    ax.set_ylabel("km/h")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 5. SOC distribution
    ax = axes[1, 0]
    if 'bcell_soc' in df.columns:
        data = [df[df.vehicle_id == v]['bcell_soc'].dropna()
                for v in vehicles]
        ax.violinplot(data, showmeans=True)
        ax.set_title("SOC Distribution")
        ax.set_ylabel("SOC (%)")
        ax.set_xticks(range(1, len(vehicles) + 1))
        ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 6. Voltage
    ax = axes[1, 1]
    ax.bar(range(len(vehicles)), stats_df['Avg Voltage'],
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Average HV Voltage (V)")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 7. Current
    ax = axes[1, 2]
    ax.bar(range(len(vehicles)), stats_df['Avg Current'],
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Average HV Current (A)")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 8. Temperature
    ax = axes[1, 3]
    ax.bar(range(len(vehicles)), stats_df['Avg Max Temp'],
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Max Cell Temperature (°C)")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # -------------------------------------------------------------------------
    # 9. Cell imbalance
    ax = axes[2, 0]
    ax.bar(range(len(vehicles)), stats_df['Cell Imbalance'],
           color=plt.cm.tab20(np.arange(len(vehicles))))
    ax.set_title("Cell Imbalance")
    ax.set_xticks(range(len(vehicles)))
    ax.set_xticklabels([f"V{v}" for v in vehicles])

    # Hide unused plots
    for i in range(2, 4):
        for j in range(1, 4):
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("vehicle_comparison_overview.png", dpi=300)
    plt.show()

    print("✅ Figure saved as 'vehicle_comparison_overview.png'")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    df = load_cleaned_data()
    stats_df = compare_vehicle_statistics(df)
    visualize_vehicle_comparison(df, stats_df)
