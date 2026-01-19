"""
SOH Distribution Analysis
Analyze State of Health distribution across vehicles and conditions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.gridspec as gridspec
plt.style.use('seaborn-v0_8-darkgrid')

def load_cleaned_data():
    """Load the cleaned dataset"""
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    df = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet")
    return df

def analyze_overall_soh_distribution(df):
    """Analyze overall SOH distribution"""
    print("="*80)
    print("OVERALL SOH DISTRIBUTION ANALYSIS")
    print("="*80)
    
    if 'soh_capacity' not in df.columns:
        print("❌ SOH column not found in dataset!")
        return None
    
    soh_data = df['soh_capacity'].dropna()
    
    # Basic statistics
    stats_dict = {
        'Count': len(soh_data),
        'Mean': soh_data.mean(),
        'Median': soh_data.median(),
        'Std': soh_data.std(),
        'Min': soh_data.min(),
        'Max': soh_data.max(),
        '25th Percentile': soh_data.quantile(0.25),
        '75th Percentile': soh_data.quantile(0.75),
        'Skewness': soh_data.skew(),
        'Kurtosis': soh_data.kurtosis()
    }
    
    print("\nSOH Statistics:")
    for stat, value in stats_dict.items():
        print(f"{stat:20}: {value:.2f}" + ("%" if stat not in ['Count', 'Skewness', 'Kurtosis'] else ""))
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Histogram with KDE
    ax1 = fig.add_subplot(gs[0, :2])
    n, bins, patches = ax1.hist(soh_data, bins=50, edgecolor='black', alpha=0.7, density=True, 
                                color='steelblue', label='Histogram')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(soh_data)
    x_range = np.linspace(soh_data.min(), soh_data.max(), 1000)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Kernel Density')
    
    ax1.axvline(soh_data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {soh_data.mean():.1f}%')
    ax1.axvline(soh_data.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {soh_data.median():.1f}%')
    
    # Highlight boundaries
    ax1.axvspan(65, 75, alpha=0.2, color='orange', label='Lower boundary zone')
    ax1.axvspan(105, 110, alpha=0.2, color='orange', label='Upper boundary zone')
    
    ax1.set_xlabel('SOH (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('SOH Distribution with Kernel Density Estimation', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = fig.add_subplot(gs[0, 2])
    box_data = ax2.boxplot(soh_data, vert=True, patch_artist=True, showfliers=False)
    box_data['boxes'][0].set_facecolor('lightblue')
    box_data['boxes'][0].set_alpha(0.7)
    
    # Add mean point
    ax2.plot(1, soh_data.mean(), 'rD', markersize=10, label=f'Mean: {soh_data.mean():.1f}%')
    
    ax2.set_ylabel('SOH (%)', fontsize=12)
    ax2.set_title('SOH Box Plot (Outliers Hidden)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot
    ax3 = fig.add_subplot(gs[1, 0])
    violin_parts = ax3.violinplot(soh_data, vert=True, showmeans=True, showmedians=True)
    violin_parts['bodies'][0].set_facecolor('lightcoral')
    violin_parts['bodies'][0].set_alpha(0.7)
    violin_parts['bodies'][0].set_edgecolor('black')
    
    ax3.set_ylabel('SOH (%)', fontsize=12)
    ax3.set_title('SOH Violin Plot', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. ECDF plot
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_soh = np.sort(soh_data)
    y_vals = np.arange(1, len(sorted_soh) + 1) / len(sorted_soh)
    ax4.plot(sorted_soh, y_vals, 'b-', linewidth=2)
    
    # Add quartile lines
    for q in [0.25, 0.5, 0.75]:
        q_value = np.percentile(soh_data, q * 100)
        ax4.axvline(q_value, color='red', linestyle='--', alpha=0.5)
        ax4.text(q_value, 0.05, f'Q{q}: {q_value:.1f}%', rotation=90, fontsize=9)
    
    ax4.set_xlabel('SOH (%)', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Empirical Cumulative Distribution', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Q plot
    ax5 = fig.add_subplot(gs[1, 2])
    stats.probplot(soh_data, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot vs Normal Distribution', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical summary table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = []
    colors = []
    for stat, value in stats_dict.items():
        if stat == 'Count':
            table_data.append([stat, f"{value:,}"])
        else:
            table_data.append([stat, f"{value:.2f}" + ("%" if stat not in ['Skewness', 'Kurtosis'] else "")])
        
        # Color coding
        if stat in ['Mean', 'Median'] and value > 90:
            colors.append(['lightgreen', 'lightgreen'])
        elif stat in ['Skewness'] and abs(value) > 1:
            colors.append(['lightcoral', 'lightcoral'])
        else:
            colors.append(['white', 'white'])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.3],
                     cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax6.set_title('SOH Statistical Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('soh_overall_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_dict

def analyze_soh_by_vehicle(df):
    """Analyze SOH distribution by vehicle"""
    print("\n" + "="*80)
    print("SOH DISTRIBUTION BY VEHICLE")
    print("="*80)
    
    if 'vehicle_id' not in df.columns or 'soh_capacity' not in df.columns:
        print("❌ Required columns not found!")
        return None
    
    # Calculate statistics by vehicle
    vehicle_stats = df.groupby('vehicle_id')['soh_capacity'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    
    vehicle_stats['range'] = vehicle_stats['max'] - vehicle_stats['min']
    vehicle_stats['cv'] = (vehicle_stats['std'] / vehicle_stats['mean'] * 100).round(2)
    vehicle_stats['iqr'] = df.groupby('vehicle_id')['soh_capacity'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).round(2)
    
    print("\nSOH Statistics by Vehicle:")
    print(vehicle_stats.to_string())
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SOH Analysis by Vehicle', fontsize=16, fontweight='bold')
    
    vehicles = vehicle_stats.index.tolist()
    colors = plt.cm.tab20(np.arange(len(vehicles)))
    
    # 1. Mean SOH by vehicle
    ax1 = axes[0, 0]
    means = vehicle_stats['mean'].values
    
    bars = ax1.bar(range(len(vehicles)), means, color=colors, edgecolor='black')
    ax1.set_xlabel('Vehicle ID', fontsize=12)
    ax1.set_ylabel('Mean SOH (%)', fontsize=12)
    ax1.set_title('Mean SOH by Vehicle', fontsize=14)
    ax1.set_xticks(range(len(vehicles)))
    ax1.set_xticklabels([f'V{vid}' for vid in vehicles])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add error bars
    ax1.errorbar(range(len(vehicles)), means, yerr=vehicle_stats['std'].values, 
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Highlight Vehicle 7 (low SOH)
    if 7 in vehicles:
        idx = vehicles.index(7)
        bars[idx].set_edgecolor('red')
        bars[idx].set_linewidth(3)
        ax1.text(idx, means[idx] - 5, 'Degraded', ha='center', color='red', fontweight='bold')
    
    # 2. Box plot by vehicle
    ax2 = axes[0, 1]
    soh_by_vehicle = [df[df['vehicle_id'] == vid]['soh_capacity'].dropna().values 
                      for vid in vehicles]
    
    box = ax2.boxplot(soh_by_vehicle, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax2.set_xlabel('Vehicle ID', fontsize=12)
    ax2.set_ylabel('SOH (%)', fontsize=12)
    ax2.set_title('SOH Distribution by Vehicle', fontsize=14)
    ax2.set_xticklabels([f'V{vid}' for vid in vehicles])
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot by vehicle
    ax3 = axes[0, 2]
    violin_parts = ax3.violinplot(soh_by_vehicle, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    ax3.set_xlabel('Vehicle ID', fontsize=12)
    ax3.set_ylabel('SOH (%)', fontsize=12)
    ax3.set_title('SOH Density by Vehicle', fontsize=14)
    ax3.set_xticks(range(1, len(vehicles) + 1))
    ax3.set_xticklabels([f'V{vid}' for vid in vehicles])
    ax3.grid(True, alpha=0.3)
    
    # 4. SOH variability by vehicle
    ax4 = axes[1, 0]
    x_pos = np.arange(len(vehicles))
    width = 0.35
    
    ax4.bar(x_pos - width/2, vehicle_stats['range'], width, 
            label='Range (Max-Min)', alpha=0.7, color='skyblue', edgecolor='black')
    ax4.bar(x_pos + width/2, vehicle_stats['cv'], width, 
            label='CV (%)', alpha=0.7, color='lightcoral', edgecolor='black')
    
    ax4.set_xlabel('Vehicle ID', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('SOH Variability by Vehicle', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'V{vid}' for vid in vehicles])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. SOH vs Vehicle Age (using mileage as proxy)
    ax5 = axes[1, 1]
    if 'vhc_totalMile' in df.columns:
        avg_mileage = df.groupby('vehicle_id')['vhc_totalMile'].mean()
        
        scatter = ax5.scatter(avg_mileage.values, vehicle_stats['mean'].values, 
                            s=100, c=colors, edgecolors='black', alpha=0.7)
        
        # Add vehicle labels
        for i, (mile, soh, vid) in enumerate(zip(avg_mileage.values, vehicle_stats['mean'].values, vehicles)):
            ax5.text(mile, soh, f' V{vid}', fontsize=9, va='center')
        
        # Add trend line
        z = np.polyfit(avg_mileage.values, vehicle_stats['mean'].values, 1)
        p = np.poly1d(z)
        ax5.plot(np.sort(avg_mileage.values), p(np.sort(avg_mileage.values)), 
                'r--', alpha=0.7, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')
        
        ax5.set_xlabel('Average Mileage (km)', fontsize=12)
        ax5.set_ylabel('Mean SOH (%)', fontsize=12)
        ax5.set_title('SOH vs Vehicle Mileage', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. SOH histogram by vehicle
    ax6 = axes[1, 2]
    for i, (vid, color) in enumerate(zip(vehicles, colors)):
        vehicle_soh = df[df['vehicle_id'] == vid]['soh_capacity'].dropna()
        ax6.hist(vehicle_soh, bins=30, alpha=0.5, color=color, 
                label=f'V{vid}', density=True, histtype='stepfilled')
    
    ax6.set_xlabel('SOH (%)', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12)
    ax6.set_title('SOH Distribution Comparison', fontsize=14)
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soh_by_vehicle_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return vehicle_stats

def analyze_soh_by_chemistry(df):
    """Analyze SOH distribution by battery chemistry"""
    print("\n" + "="*80)
    print("SOH DISTRIBUTION BY CHEMISTRY")
    print("="*80)
    
    if 'chemistry' not in df.columns or 'soh_capacity' not in df.columns:
        print("❌ Required columns not found!")
        return None
    
    # Calculate statistics by chemistry
    chem_stats = df.groupby('chemistry')['soh_capacity'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median', 
        lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
    ]).round(2)
    
    chem_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    chem_stats['iqr'] = chem_stats['q75'] - chem_stats['q25']
    
    print("\nSOH Statistics by Chemistry:")
    print(chem_stats.to_string())
    
    # Statistical test for difference between chemistries
    chemistries = df['chemistry'].unique()
    if len(chemistries) >= 2:
        soh_by_chem = [df[df['chemistry'] == chem]['soh_capacity'].dropna().values 
                       for chem in chemistries]
        
        # T-test for two groups
        if len(chemistries) == 2:
            t_stat, p_value = stats.ttest_ind(*soh_by_chem, equal_var=False)
            print(f"\nT-test between {chemistries[0]} and {chemistries[1]}:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4e}")
            
            significance = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT SIGNIFICANT"
            print(f"  {significance} (p < 0.05)")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SOH Analysis by Battery Chemistry', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for NCM, Orange for LFP
    
    # 1. Mean SOH by chemistry
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(chemistries)), chem_stats['mean'], color=colors, edgecolor='black', alpha=0.7)
    
    ax1.set_xlabel('Chemistry', fontsize=12)
    ax1.set_ylabel('Mean SOH (%)', fontsize=12)
    ax1.set_title('Mean SOH by Battery Chemistry', fontsize=14)
    ax1.set_xticks(range(len(chemistries)))
    ax1.set_xticklabels(chemistries)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add error bars
    ax1.errorbar(range(len(chemistries)), chem_stats['mean'], 
                yerr=chem_stats['std'], fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    
    # Add significance star if p < 0.05
    if 'p_value' in locals() and p_value < 0.05:
        ax1.text(0.5, max(chem_stats['mean']) + 5, f'p = {p_value:.2e}', 
                ha='center', fontweight='bold', color='red')
    
    # 2. Box plot by chemistry
    ax2 = axes[0, 1]
    soh_data_by_chem = [df[df['chemistry'] == chem]['soh_capacity'].dropna().values 
                        for chem in chemistries]
    
    box = ax2.boxplot(soh_data_by_chem, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax2.set_xlabel('Chemistry', fontsize=12)
    ax2.set_ylabel('SOH (%)', fontsize=12)
    ax2.set_title('SOH Distribution by Chemistry', fontsize=14)
    ax2.set_xticklabels(chemistries)
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot by chemistry
    ax3 = axes[0, 2]
    violin_parts = ax3.violinplot(soh_data_by_chem, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    ax3.set_xlabel('Chemistry', fontsize=12)
    ax3.set_ylabel('SOH (%)', fontsize=12)
    ax3.set_title('SOH Density by Chemistry', fontsize=14)
    ax3.set_xticks(range(1, len(chemistries) + 1))
    ax3.set_xticklabels(chemistries)
    ax3.grid(True, alpha=0.3)
    
    # 4. ECDF comparison
    ax4 = axes[1, 0]
    for i, (chem, color) in enumerate(zip(chemistries, colors)):
        chem_data = df[df['chemistry'] == chem]['soh_capacity'].dropna().sort_values()
        y_vals = np.arange(1, len(chem_data) + 1) / len(chem_data)
        ax4.plot(chem_data, y_vals, color=color, linewidth=2, label=chem)
    
    ax4.set_xlabel('SOH (%)', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution by Chemistry', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Chemistry composition
    ax5 = axes[1, 1]
    chem_counts = df['chemistry'].value_counts()
    wedges, texts, autotexts = ax5.pie(chem_counts.values, colors=colors, autopct='%1.1f%%',
                                       startangle=90, wedgeprops=dict(edgecolor='black'))
    
    ax5.set_title('Dataset Composition by Chemistry', fontsize=14)
    
    # 6. Statistical summary table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = []
    for chem in chemistries:
        stats_row = chem_stats.loc[chem]
        table_data.append([
            chem,
            f"{stats_row['count']:,}",
            f"{stats_row['mean']:.1f}%",
            f"{stats_row['std']:.1f}%",
            f"{stats_row['min']:.1f}%",
            f"{stats_row['max']:.1f}%",
            f"{stats_row['median']:.1f}%",
            f"{stats_row['iqr']:.1f}%"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Chem', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Median', 'IQR'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig('soh_by_chemistry_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return chem_stats

def analyze_soh_by_vehicle_type(df):
    """Analyze SOH distribution by vehicle type"""
    print("\n" + "="*80)
    print("SOH DISTRIBUTION BY VEHICLE TYPE")
    print("="*80)
    
    if 'vehicle_type' not in df.columns or 'soh_capacity' not in df.columns:
        print("❌ Required columns not found!")
        return None
    
    # Calculate statistics by vehicle type
    type_stats = df.groupby('vehicle_type')['soh_capacity'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    
    print("\nSOH Statistics by Vehicle Type:")
    print(type_stats.to_string())
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    types = type_stats.index.tolist()
    colors = ['skyblue', 'lightcoral']
    
    # 1. Bar chart
    ax1 = axes[0, 0]
    means = type_stats['mean'].values
    
    bars = ax1.bar(types, means, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Vehicle Type', fontsize=12)
    ax1.set_ylabel('Mean SOH (%)', fontsize=12)
    ax1.set_title('Mean SOH by Vehicle Type', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax1.errorbar(types, means, yerr=type_stats['std'].values, 
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    soh_by_type = [df[df['vehicle_type'] == vt]['soh_capacity'].dropna().values 
                   for vt in types]
    
    box = ax2.boxplot(soh_by_type, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    ax2.set_xlabel('Vehicle Type', fontsize=12)
    ax2.set_ylabel('SOH (%)', fontsize=12)
    ax2.set_title('SOH Distribution by Vehicle Type', fontsize=14)
    ax2.set_xticklabels(types)
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot
    ax3 = axes[0, 2]
    violin_parts = ax3.violinplot(soh_by_type, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    ax3.set_xlabel('Vehicle Type', fontsize=12)
    ax3.set_ylabel('SOH (%)', fontsize=12)
    ax3.set_title('SOH Density by Vehicle Type', fontsize=14)
    ax3.set_xticks(range(1, len(types) + 1))
    ax3.set_xticklabels(types)
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = axes[1, 0]
    for i, (vt, color) in enumerate(zip(types, colors)):
        type_data = df[df['vehicle_type'] == vt]['soh_capacity'].dropna()
        ax4.hist(type_data, bins=30, alpha=0.5, color=color, 
                label=vt, density=True, histtype='stepfilled')
    
    ax4.set_xlabel('SOH (%)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('SOH Distribution Comparison', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. ECDF comparison
    ax5 = axes[1, 1]
    for i, (vt, color) in enumerate(zip(types, colors)):
        type_data = df[df['vehicle_type'] == vt]['soh_capacity'].dropna().sort_values()
        y_vals = np.arange(1, len(type_data) + 1) / len(type_data)
        ax5.plot(type_data, y_vals, color=color, linewidth=2, label=vt)
    
    ax5.set_xlabel('SOH (%)', fontsize=12)
    ax5.set_ylabel('Cumulative Probability', fontsize=12)
    ax5.set_title('Cumulative Distribution by Type', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Data composition
    ax6 = axes[1, 2]
    type_counts = df['vehicle_type'].value_counts()
    wedges, texts, autotexts = ax6.pie(type_counts.values, colors=colors, autopct='%1.1f%%',
                                       startangle=90, wedgeprops=dict(edgecolor='black'))
    
    ax6.set_title('Dataset Composition by Vehicle Type', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('soh_by_vehicle_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return type_stats

def analyze_soh_correlation_with_features(df):
    """Analyze correlation between SOH and various features"""
    print("\n" + "="*80)
    print("SOH CORRELATION WITH FEATURES")
    print("="*80)
    
    # Select features to analyze
    features = [
        'vhc_totalMile', 'bcell_soc', 'hv_voltage', 'hv_current',
        'bcell_maxVoltage', 'bcell_minVoltage', 'cell_imbalance',
        'bcell_maxTemp', 'bcell_minTemp', 'temp_gradient', 'power_kw'
    ]
    
    available_features = [feat for feat in features if feat in df.columns]
    
    if not available_features or 'soh_capacity' not in df.columns:
        print("❌ Required columns not found!")
        return None
    
    # Calculate correlations
    correlations = []
    for feat in available_features:
        # Remove rows with missing values for this pair
        valid_data = df[[feat, 'soh_capacity']].dropna()
        if len(valid_data) > 100:  # Need sufficient data
            corr = valid_data[feat].corr(valid_data['soh_capacity'])
            correlations.append((feat, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nFeature Correlations with SOH:")
    for feat, corr in correlations:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"{feat:20}: {corr:.4f} ({direction} {strength})")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    corr_data = df[available_features + ['soh_capacity']].corr()
    soh_corrs = corr_data['soh_capacity'].drop('soh_capacity').sort_values()
    
    im = ax1.imshow(np.array([soh_corrs.values]).T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_yticks(range(len(soh_corrs)))
    ax1.set_yticklabels(soh_corrs.index)
    ax1.set_title('Feature Correlations with SOH', fontsize=14)
    
    # Add correlation values
    for i, (feat, corr) in enumerate(soh_corrs.items()):
        color = 'white' if abs(corr) > 0.5 else 'black'
        ax1.text(0, i, f'{corr:.3f}', ha='center', va='center', color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # 2. Top correlations bar chart
    ax2 = axes[0, 1]
    top_n = min(10, len(correlations))
    top_features = [feat for feat, _ in correlations[:top_n]]
    top_corrs = [corr for _, corr in correlations[:top_n]]
    
    bars = ax2.barh(range(top_n), top_corrs, color=np.where(np.array(top_corrs) > 0, 'green', 'red'))
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_title(f'Top {top_n} Feature Correlations with SOH', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
        ax2.text(corr + (0.01 if corr >= 0 else -0.05), i, f'{corr:.3f}', 
                va='center', fontsize=9, color='black' if abs(corr) < 0.7 else 'white')
    
    # 3. Scatter plot for top correlated feature
    ax3 = axes[1, 0]
    if correlations:
        top_feat, top_corr = correlations[0]
        ax3.scatter(df[top_feat], df['soh_capacity'], alpha=0.3, s=10, color='steelblue')
        ax3.set_xlabel(top_feat)
        ax3.set_ylabel('SOH (%)')
        ax3.set_title(f'SOH vs {top_feat} (Corr: {top_corr:.3f})', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        valid_data = df[[top_feat, 'soh_capacity']].dropna()
        if len(valid_data) > 10:
            z = np.polyfit(valid_data[top_feat], valid_data['soh_capacity'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(valid_data[top_feat].min(), valid_data[top_feat].max(), 100)
            ax3.plot(x_range, p(x_range), 'r--', linewidth=2)
    
    # 4. Correlation matrix (top features only)
    ax4 = axes[1, 1]
    top_feat_list = [feat for feat, _ in correlations[:min(8, len(correlations))]]
    if top_feat_list:
        corr_matrix = df[top_feat_list + ['soh_capacity']].corr()
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.columns)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax4.set_yticklabels(corr_matrix.columns)
        ax4.set_title('Correlation Matrix (Top Features)', fontsize=14)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black',
                        fontsize=8)
        
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('soh_feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(correlations)

def main():
    """Main SOH analysis function"""
    print("STARTING COMPREHENSIVE SOH DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    # Perform analyses
    overall_stats = analyze_overall_soh_distribution(df)
    vehicle_stats = analyze_soh_by_vehicle(df)
    chem_stats = analyze_soh_by_chemistry(df)
    type_stats = analyze_soh_by_vehicle_type(df)
    feature_corrs = analyze_soh_correlation_with_features(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Generate summary
    print("\n📊 KEY FINDINGS:")
    
    if overall_stats:
        print(f"1. Overall SOH: Mean = {overall_stats.get('Mean', 'N/A'):.1f}%, Range = {overall_stats.get('Min', 'N/A'):.1f}%-{overall_stats.get('Max', 'N/A'):.1f}%")
    
    if vehicle_stats is not None and not vehicle_stats.empty:
        lowest_soh = vehicle_stats['mean'].idxmin()
        highest_soh = vehicle_stats['mean'].idxmax()
        print(f"2. Vehicle with lowest SOH: V{lowest_soh} ({vehicle_stats.loc[lowest_soh, 'mean']:.1f}%)")
        print(f"3. Vehicle with highest SOH: V{highest_soh} ({vehicle_stats.loc[highest_soh, 'mean']:.1f}%)")
    
    if chem_stats is not None and not chem_stats.empty:
        for chem in chem_stats.index:
            print(f"4. {chem} SOH: Mean = {chem_stats.loc[chem, 'mean']:.1f}% (n={chem_stats.loc[chem, 'count']:,})")
    
    if feature_corrs:
        top_feat = max(feature_corrs.items(), key=lambda x: abs(x[1]))
        print(f"5. Top SOH correlate: {top_feat[0]} (r = {top_feat[1]:.3f})")
    
    print("\n📈 Generated Visualizations:")
    print("1. soh_overall_distribution.png - Comprehensive SOH analysis")
    print("2. soh_by_vehicle_distribution.png - Vehicle-wise SOH analysis")
    print("3. soh_by_chemistry_distribution.png - Chemistry comparison")
    print("4. soh_by_vehicle_type_distribution.png - Vehicle type analysis")
    print("5. soh_feature_correlations.png - Feature correlations with SOH")

if __name__ == "__main__":
    main()