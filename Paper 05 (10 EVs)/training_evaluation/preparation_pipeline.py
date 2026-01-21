"""
OPTIMIZED DEEP LEARNING DATA PREPARATION FOR SOH ESTIMATION
Phase 1: Sequence Creation and Feature Engineering - OPTIMIZED VERSION
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Debug mode for faster testing
DEBUG_MODE = False  # Set to True for faster testing, False for full run

class SOHSequenceDataset(Dataset):
    """Create sequence datasets for deep learning models - OPTIMIZED VERSION"""
    
    def __init__(self, df, sequence_length=360, target_col='soh_capacity', 
                 step_size=10, mode='train', vehicle_ids=None):
        """
        Args:
            df: Cleaned dataframe
            sequence_length: Number of time steps in sequence (360 = 60 min at 10s intervals)
            target_col: Target column name
            step_size: Step between sequences (controls overlap)
            mode: 'train', 'val', or 'test'
            vehicle_ids: Specific vehicles to include
        """
        start_time = time.time()
        
        # Adjust for debug mode
        if DEBUG_MODE:
            sequence_length = min(sequence_length, 60)  # Shorter sequences for testing
            step_size = max(step_size, 30)  # Less overlap
            print("DEBUG MODE: Using reduced parameters")
        
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.step_size = step_size
        self.mode = mode
        
        # Filter by vehicle if specified
        if vehicle_ids is not None:
            self.df = df[df['vehicle_id'].isin(vehicle_ids)].copy()
        else:
            self.df = df.copy()
        
        # Sample for debugging
        if DEBUG_MODE and len(self.df) > 10000:
            self.df = self.df.sample(n=10000, random_state=42)
            print(f"DEBUG: Sampled to {len(self.df):,} rows")
        
        # Ensure session_change exists
        if 'session_change' not in self.df.columns:
            print("Warning: 'session_change' column not found. Creating it...")
            self._create_session_change()
        
        # Define feature groups
        self._define_feature_groups()
        
        # Prepare sequences
        print(f"Creating sequences for {mode} set...")
        self.sequences, self.targets, self.metadata = self._create_sequences()
        
        elapsed = time.time() - start_time
        print(f"Created {len(self.sequences)} sequences of length {sequence_length} in {elapsed:.1f}s")
        print(f"Features: {len(self.feature_cols)}")
    
    def _create_session_change(self):
        """Create session_change if it doesn't exist"""
        self.df = self.df.sort_values(['vehicle_id', 'time'])
        
        # Identify driving sessions (speed > 1 km/h)
        if 'vhc_speed' in self.df.columns:
            self.df['is_driving'] = (self.df['vhc_speed'] > 1).astype(int)
        else:
            self.df['is_driving'] = 0
        
        # Session changes when: vehicle changes OR driving state changes significantly
        self.df['session_change'] = (
            (self.df['vehicle_id'] != self.df['vehicle_id'].shift()) |
            (self.df['is_driving'].diff().abs() > 0.5)
        ).cumsum()
    
    def _define_feature_groups(self):
        """Define feature groups for deep learning"""
        
        # Group 1: Raw measurements
        self.raw_features = [
            'vhc_speed', 'hv_current', 'hv_voltage', 'bcell_soc',
            'bcell_maxVoltage', 'bcell_minVoltage', 
            'bcell_maxTemp', 'bcell_minTemp'
        ]
        
        # Group 2: Derived features (from cleaning)
        self.derived_features = [
            'cell_imbalance', 'temp_gradient', 'power_kw'
        ]
        
        # Group 3: Statistical features (to be created)
        self.stat_features = []
        
        # Combine available features
        self.feature_cols = []
        for feat_list in [self.raw_features, self.derived_features]:
            for feat in feat_list:
                if feat in self.df.columns:
                    self.feature_cols.append(feat)
        
        # Add rolling statistics - OPTIMIZED VERSION
        self._add_rolling_features_optimized()
    
    def _add_rolling_features_optimized(self):
        """Add rolling window statistics as features - OPTIMIZED VERSION"""
        
        print("Creating rolling window features (optimized)...")
        start_time = time.time()
        
        # Sort data for rolling operations
        self.df = self.df.sort_values(['vehicle_id', 'session_change', 'time']).reset_index(drop=True)
        
        # Create a group key for rolling operations
        group_key = ['vehicle_id', 'session_change']
        
        # 5-minute rolling features (30 samples at 10s intervals) - VECTORIZED
        if 'hv_current' in self.df.columns:
            print("  Creating current rolling features...")
            # Use groupby + rolling with vectorized operations
            self.df['rolling_mean_current_5min'] = (
                self.df.groupby(group_key)['hv_current']
                .rolling(window=30, min_periods=5, center=False)
                .mean()
                .reset_index(level=[0,1], drop=True)
            )
            
            self.df['rolling_std_current_5min'] = (
                self.df.groupby(group_key)['hv_current']
                .rolling(window=30, min_periods=5, center=False)
                .std()
                .reset_index(level=[0,1], drop=True)
            )
        
        if 'bcell_maxVoltage' in self.df.columns and 'bcell_minVoltage' in self.df.columns:
            print("  Creating voltage rolling features...")
            # Voltage range
            rolling_max = (
                self.df.groupby(group_key)['bcell_maxVoltage']
                .rolling(window=30, min_periods=5, center=False)
                .max()
                .reset_index(level=[0,1], drop=True)
            )
            
            rolling_min = (
                self.df.groupby(group_key)['bcell_minVoltage']
                .rolling(window=30, min_periods=5, center=False)
                .min()
                .reset_index(level=[0,1], drop=True)
            )
            
            self.df['rolling_voltage_range_5min'] = rolling_max - rolling_min
        
        if 'bcell_maxTemp' in self.df.columns:
            print("  Creating temperature rolling features...")
            self.df['temp_variance_5min'] = (
                self.df.groupby(group_key)['bcell_maxTemp']
                .rolling(window=30, min_periods=5, center=False)
                .var()
                .reset_index(level=[0,1], drop=True)
            )
        
        # Cumulative features - OPTIMIZED
        if 'hv_current' in self.df.columns:
            print("  Creating cumulative features...")
            # Create masks for charge/discharge
            charge_mask = self.df['hv_current'] > 0
            discharge_mask = self.df['hv_current'] < 0
            
            # Initialize columns with zeros
            self.df['cumulative_charge'] = 0.0
            self.df['cumulative_discharge'] = 0.0
            
            # Use vectorized groupby operations
            if charge_mask.any():
                # Create a temporary column for absolute current during charging
                self.df.loc[charge_mask, 'abs_charge_current'] = self.df.loc[charge_mask, 'hv_current'].abs()
                # Grouped cumulative sum
                self.df.loc[charge_mask, 'cumulative_charge'] = (
                    self.df.loc[charge_mask]
                    .groupby(group_key)['abs_charge_current']
                    .transform(lambda x: x.cumsum())
                )
                # Clean up temporary column
                self.df = self.df.drop('abs_charge_current', axis=1, errors='ignore')
            
            if discharge_mask.any():
                # Create a temporary column for absolute current during discharging
                self.df.loc[discharge_mask, 'abs_discharge_current'] = self.df.loc[discharge_mask, 'hv_current'].abs()
                # Grouped cumulative sum
                self.df.loc[discharge_mask, 'cumulative_discharge'] = (
                    self.df.loc[discharge_mask]
                    .groupby(group_key)['abs_discharge_current']
                    .transform(lambda x: x.cumsum())
                )
                # Clean up temporary column
                self.df = self.df.drop('abs_discharge_current', axis=1, errors='ignore')
        
        # Add statistical features to list
        self.stat_features = [
            'rolling_mean_current_5min', 'rolling_std_current_5min',
            'rolling_voltage_range_5min', 'temp_variance_5min',
            'cumulative_charge', 'cumulative_discharge'
        ]
        
        # Fill NaN values in new features efficiently
        for feat in self.stat_features:
            if feat in self.df.columns:
                # Use vectorized fillna
                self.df[feat] = self.df[feat].fillna(method='ffill', limit=10).fillna(0)
        
        # Add to feature columns if they exist
        for feat in self.stat_features:
            if feat in self.df.columns and feat not in self.feature_cols:
                self.feature_cols.append(feat)
        
        elapsed = time.time() - start_time
        print(f"  Rolling features created in {elapsed:.2f}s")
        print(f"  Added {len(self.stat_features)} statistical features")
    
    def _create_sequences_optimized(self):
        """Create overlapping sequences from time series data - OPTIMIZED"""
        
        sequences = []
        targets = []
        metadata = []
        
        # Group by vehicle and session
        grouped = self.df.groupby(['vehicle_id', 'session_change'])
        total_groups = len(grouped)
        
        print(f"Processing {total_groups} vehicle-session groups...")
        
        # Pre-collect all indices to avoid repeated .loc calls
        all_indices = []
        all_vehicle_ids = []
        all_session_ids = []
        
        for (vehicle_id, session_id), group_indices in grouped.groups.items():
            all_indices.append(group_indices)
            all_vehicle_ids.append(vehicle_id)
            all_session_ids.append(session_id)
        
        # Process in chunks to avoid memory issues
        chunk_size = 100  # Process 100 groups at a time
        for chunk_idx in range(0, len(all_indices), chunk_size):
            chunk_end = min(chunk_idx + chunk_size, len(all_indices))
            
            for i in range(chunk_idx, chunk_end):
                group_indices = all_indices[i]
                vehicle_id = all_vehicle_ids[i]
                session_id = all_session_ids[i]
                
                # Get data for this group
                group = self.df.loc[group_indices].copy()
                
                # Sort by time if needed
                if 'time' in group.columns:
                    group = group.sort_values('time')
                
                # Ensure we have target values
                if self.target_col not in group.columns or group[self.target_col].isna().all():
                    continue
                
                # Get feature matrix - ensure all features exist
                available_features = [f for f in self.feature_cols if f in group.columns]
                feature_data = group[available_features].values
                target_data = group[self.target_col].values
                
                # Skip if not enough data for one sequence
                if len(feature_data) < self.sequence_length:
                    continue
                
                # Create sequences with sliding window
                max_start = len(feature_data) - self.sequence_length
                
                for start_idx in range(0, max_start + 1, self.step_size):
                    end_idx = start_idx + self.sequence_length
                    
                    # Check if we have valid target at the end of sequence
                    if not np.isnan(target_data[end_idx - 1]):
                        sequence = feature_data[start_idx:end_idx]
                        
                        # Quick NaN check
                        nan_ratio = np.isnan(sequence).mean()
                        if nan_ratio < 0.3:  # Less than 30% NaN (relaxed)
                            sequences.append(sequence)
                            targets.append(target_data[end_idx - 1])
                            metadata.append({
                                'vehicle_id': vehicle_id,
                                'session_id': session_id,
                                'start_idx': start_idx,
                                'end_idx': end_idx
                            })
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences created! Check your data and parameters.")
        
        # Convert to numpy arrays
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        # Handle NaN values in sequences efficiently
        print("Handling NaN values in sequences...")
        
        # Vectorized NaN handling
        nan_mask = np.isnan(sequences_array)
        if nan_mask.any():
            # Create a copy for processing
            seq_clean = sequences_array.copy()
            
            # Forward fill along time axis (axis=1)
            for i in range(1, seq_clean.shape[1]):
                current_nan = np.isnan(seq_clean[:, i, :])
                prev_not_nan = ~np.isnan(seq_clean[:, i-1, :])
                fill_mask = current_nan & prev_not_nan
                seq_clean[:, i, :][fill_mask] = seq_clean[:, i-1, :][fill_mask]
            
            # Backward fill along time axis
            for i in range(seq_clean.shape[1]-2, -1, -1):
                current_nan = np.isnan(seq_clean[:, i, :])
                next_not_nan = ~np.isnan(seq_clean[:, i+1, :])
                fill_mask = current_nan & next_not_nan
                seq_clean[:, i, :][fill_mask] = seq_clean[:, i+1, :][fill_mask]
            
            # Fill remaining with zeros
            seq_clean[np.isnan(seq_clean)] = 0
            sequences_array = seq_clean
        
        return sequences_array, targets_array, metadata
    
    def _create_sequences(self):
        """Wrapper for sequence creation - uses optimized version"""
        return self._create_sequences_optimized()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        return {
            'sequence': sequence,
            'target': target,
            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
        }
    
    def get_feature_names(self):
        return self.feature_cols


class DataScaler:
    """Handle scaling and normalization of sequences - OPTIMIZED VERSION"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scalers = {}  # One scaler per feature
        self.feature_stats = {}  # Store statistics for each feature
        
    def fit(self, sequences):
        """Fit scalers on training data"""
        # sequences shape: (n_samples, seq_len, n_features)
        n_features = sequences.shape[2]
        
        print(f"Fitting scalers for {n_features} features...")
        start_time = time.time()
        
        # Flatten sequences for batch processing
        sequences_flat = sequences.reshape(-1, n_features)
        
        for feature_idx in range(n_features):
            # Extract values for this feature
            feature_values = sequences_flat[:, feature_idx].reshape(-1, 1)
            
            # Remove NaN values
            non_nan_mask = ~np.isnan(feature_values.flatten())
            clean_values = feature_values[non_nan_mask].reshape(-1, 1)
            
            if len(clean_values) == 0:
                # If all values are NaN, use standard scaler with zeros
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                    scaler.mean_ = np.array([0])
                    scaler.scale_ = np.array([1])
                else:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler.data_min_ = np.array([0])
                    scaler.data_max_ = np.array([1])
            else:
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                elif self.scaler_type == 'minmax':
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                else:
                    raise ValueError(f"Unknown scaler type: {self.scaler_type}")
                
                scaler.fit(clean_values)
            
            self.scalers[feature_idx] = scaler
            
            # Store statistics
            self.feature_stats[feature_idx] = {
                'mean': float(np.mean(clean_values) if len(clean_values) > 0 else 0),
                'std': float(np.std(clean_values) if len(clean_values) > 0 else 1),
                'min': float(np.min(clean_values) if len(clean_values) > 0 else 0),
                'max': float(np.max(clean_values) if len(clean_values) > 0 else 1),
                'n_samples': len(clean_values)
            }
        
        elapsed = time.time() - start_time
        print(f"Scalers fitted in {elapsed:.2f}s")
        return self
    
    def transform(self, sequences):
        """Transform sequences using fitted scalers - OPTIMIZED"""
        n_samples, seq_len, n_features = sequences.shape
        
        # Reshape to 2D for batch transformation
        sequences_2d = sequences.reshape(-1, n_features)
        transformed_2d = np.zeros_like(sequences_2d)
        
        for feature_idx in range(n_features):
            scaler = self.scalers.get(feature_idx)
            if scaler:
                # Extract and transform this feature
                feature_values = sequences_2d[:, feature_idx].reshape(-1, 1)
                
                # Handle NaN values
                nan_mask = np.isnan(feature_values.flatten())
                feature_values_clean = feature_values.copy()
                
                # Replace NaN with feature mean for transformation
                if np.any(nan_mask):
                    feature_mean = self.feature_stats[feature_idx]['mean']
                    feature_values_clean[nan_mask] = feature_mean
                
                # Transform
                transformed_values = scaler.transform(feature_values_clean)
                
                # Put NaN back if they were there
                if np.any(nan_mask):
                    transformed_values = transformed_values.reshape(-1)
                    transformed_values[nan_mask] = np.nan
                    transformed_values = transformed_values.reshape(-1, 1)
                
                transformed_2d[:, feature_idx] = transformed_values.flatten()
        
        # Reshape back to 3D
        transformed = transformed_2d.reshape(n_samples, seq_len, n_features)
        return transformed
    
    def save(self, path):
        """Save scalers to disk"""
        joblib.dump({
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'scaler_type': self.scaler_type
        }, path)
    
    def load(self, path):
        """Load scalers from disk"""
        data = joblib.load(path)
        self.scalers = data['scalers']
        self.feature_stats = data.get('feature_stats', {})
        self.scaler_type = data['scaler_type']
        return self


def prepare_dataloaders(df, config):
    """
    Prepare train/val/test dataloaders with proper splitting - OPTIMIZED VERSION
    """
    
    # Configuration
    seq_length = config.get('sequence_length', 360)
    batch_size = config.get('batch_size', 32)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    random_seed = config.get('random_seed', 42)
    
    # Reduce parameters for debug mode
    if DEBUG_MODE:
        seq_length = min(seq_length, 60)
        batch_size = min(batch_size, 16)
        print("DEBUG: Using reduced sequence length and batch size")
    
    # Ensure target column exists
    if 'soh_capacity' not in df.columns:
        raise ValueError("'soh_capacity' column not found in data!")
    
    # Split vehicle IDs for proper cross-validation
    vehicle_ids = df['vehicle_id'].unique()
    
    # Strategy 1: Leave one vehicle out for test
    np.random.seed(random_seed)
    np.random.shuffle(vehicle_ids)
    
    n_test = max(1, int(len(vehicle_ids) * test_ratio))
    test_vehicles = vehicle_ids[:n_test]
    remaining_vehicles = vehicle_ids[n_test:]
    
    # Split remaining into train/val
    n_val = max(1, int(len(remaining_vehicles) * val_ratio))
    val_vehicles = remaining_vehicles[:n_val]
    train_vehicles = remaining_vehicles[n_val:]
    
    print(f"\nVehicle Splits:")
    print(f"  Train vehicles ({len(train_vehicles)}): {sorted(train_vehicles)}")
    print(f"  Val vehicles ({len(val_vehicles)}): {sorted(val_vehicles)}")
    print(f"  Test vehicles ({len(test_vehicles)}): {sorted(test_vehicles)}")
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = SOHSequenceDataset(
        df[df['vehicle_id'].isin(train_vehicles)],
        sequence_length=seq_length,
        mode='train'
    )
    
    print("\nCreating validation dataset...")
    val_dataset = SOHSequenceDataset(
        df[df['vehicle_id'].isin(val_vehicles)],
        sequence_length=seq_length,
        mode='val'
    )
    
    print("\nCreating test dataset...")
    test_dataset = SOHSequenceDataset(
        df[df['vehicle_id'].isin(test_vehicles)],
        sequence_length=seq_length,
        mode='test'
    )
    
    # Fit scaler on training data only
    print("\nFitting scaler on training data...")
    scaler = DataScaler(scaler_type='standard')
    scaler.fit(train_dataset.sequences)
    
    # Apply scaling to all datasets
    print("Applying scaling to datasets...")
    train_dataset.sequences = scaler.transform(train_dataset.sequences)
    val_dataset.sequences = scaler.transform(val_dataset.sequences)
    test_dataset.sequences = scaler.transform(test_dataset.sequences)
    
    # Save scaler
    scaler.save('data_scaler.pkl')
    print(f"Scaler saved to data_scaler.pkl")
    
    # Create dataloaders
    num_workers = config.get('num_workers', 0)  # Set to 0 for debugging, 2-4 for production
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    # Print dataset statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    print(f"Sequence length: {seq_length} timesteps")
    print(f"Features per timestep: {len(train_dataset.feature_cols)}")
    print(f"Batch size: {batch_size}")
    
    # Show target statistics
    print(f"\nTarget (SOH) Statistics:")
    print(f"  Training:   Mean = {train_dataset.targets.mean():.2f}%, Std = {train_dataset.targets.std():.2f}%, Range = [{train_dataset.targets.min():.2f}, {train_dataset.targets.max():.2f}]")
    print(f"  Validation: Mean = {val_dataset.targets.mean():.2f}%, Std = {val_dataset.targets.std():.2f}%, Range = [{val_dataset.targets.min():.2f}, {val_dataset.targets.max():.2f}]")
    print(f"  Test:       Mean = {test_dataset.targets.mean():.2f}%, Std = {test_dataset.targets.std():.2f}%, Range = [{test_dataset.targets.min():.2f}, {test_dataset.targets.max():.2f}]")
    
    # Show feature names
    print("\nFeatures used:")
    feature_names = train_dataset.get_feature_names()
    for i, feat in enumerate(feature_names):
        print(f"  {i:2d}. {feat}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'scaler': scaler,
        'feature_names': feature_names,
        'vehicle_splits': {
            'train': train_vehicles,
            'val': val_vehicles,
            'test': test_vehicles
        }
    }


def visualize_sequences(dataloaders, num_samples=3):
    """Visualize sample sequences for debugging"""
    
    train_loader = dataloaders['train_loader']
    feature_names = dataloaders['feature_names']
    
    # Get a batch
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        print("Warning: Train loader is empty!")
        return
    
    sequences = batch['sequence']
    targets = batch['target']
    
    print(f"Batch shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample targets: {targets[:5].flatten()}")
    
    # Plot sample sequences
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Select random features to plot
    n_features = sequences.shape[2]
    if n_features < 8:
        feature_indices = list(range(n_features))
    else:
        feature_indices = np.random.choice(n_features, 8, replace=False)
    
    for idx, (ax, feat_idx) in enumerate(zip(axes, feature_indices)):
        if idx >= len(feature_indices):
            ax.axis('off')
            continue
            
        # Plot first few samples
        for sample_idx in range(min(3, sequences.shape[0])):
            ax.plot(sequences[sample_idx, :, feat_idx].numpy(), 
                   alpha=0.7, linewidth=1, label=f'Sample {sample_idx}' if idx == 0 else "")
        
        ax.set_title(f"Feature: {feature_names[feat_idx]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    
    plt.suptitle("Sample Sequences (Multiple Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sequence_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot target distribution
    all_targets = []
    try:
        for batch in train_loader:
            all_targets.extend(batch['target'].numpy().flatten())
            if len(all_targets) > 10000:  # Limit for performance
                break
    except:
        print("Warning: Could not collect targets from train loader")
    
    if all_targets:
        plt.figure(figsize=(10, 6))
        plt.hist(all_targets, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.xlabel('SOH (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of SOH Targets in Training Set')
        plt.grid(True, alpha=0.3)
        plt.savefig('soh_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()


def analyze_data_quality(df):
    """Analyze data quality before processing"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    
    print("\nMissing Values (Top 10):")
    for col in missing.sort_values(ascending=False).head(10).index:
        print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
    
    # Target variable analysis
    if 'soh_capacity' in df.columns:
        print(f"\nSOH Statistics:")
        print(f"  Mean: {df['soh_capacity'].mean():.2f}%")
        print(f"  Std: {df['soh_capacity'].std():.2f}%")
        print(f"  Min: {df['soh_capacity'].min():.2f}%")
        print(f"  Max: {df['soh_capacity'].max():.2f}%")
        print(f"  Missing: {df['soh_capacity'].isnull().sum():,}")
    
    # Vehicle distribution
    if 'vehicle_id' in df.columns:
        print(f"\nVehicle Distribution:")
        vehicle_counts = df['vehicle_id'].value_counts()
        for vehicle_id, count in vehicle_counts.items():
            print(f"  Vehicle {vehicle_id}: {count:,} rows ({count/len(df)*100:.1f}%)")
    
    # Session analysis
    if 'session_change' in df.columns:
        print(f"\nSession Analysis:")
        print(f"  Total sessions: {df['session_change'].nunique():,}")
        session_lengths = df.groupby('session_change').size()
        print(f"  Avg session length: {session_lengths.mean():.1f} rows")
        print(f"  Min session length: {session_lengths.min():.1f} rows")
        print(f"  Max session length: {session_lengths.max():.1f} rows")


def main_data_preparation():
    """Main data preparation pipeline - OPTIMIZED VERSION"""
    
    total_start_time = time.time()
    print("="*80)
    print("PHASE 1: OPTIMIZED DEEP LEARNING DATA PREPARATION")
    print("="*80)
    
    # Load cleaned data with selective columns to save memory
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    
    # Define essential columns
    essential_cols = [
        'vehicle_id', 'soh_capacity', 'vhc_speed', 'hv_current', 
        'hv_voltage', 'bcell_soc', 'bcell_maxVoltage', 'bcell_minVoltage',
        'bcell_maxTemp', 'bcell_minTemp', 'time', 'session_change',
        'cell_imbalance', 'temp_gradient', 'power_kw'
    ]
    
    print("Loading essential columns from cleaned data...")
    load_start = time.time()
    
    # Try to load only essential columns
    try:
        df_cleaned = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet", 
                                    columns=essential_cols)
    except:
        # If column selection fails, load all and select
        print("Column selection failed, loading all columns...")
        df_cleaned = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet")
        df_cleaned = df_cleaned[essential_cols]
    
    load_time = time.time() - load_start
    print(f"Data loaded in {load_time:.2f}s: {len(df_cleaned):,} rows × {len(df_cleaned.columns)} columns")
    
    # Sample for debug mode
    if DEBUG_MODE and len(df_cleaned) > 50000:
        df_cleaned = df_cleaned.sample(n=50000, random_state=42)
        print(f"DEBUG: Sampled to {len(df_cleaned):,} rows")
    
    # Analyze data quality
    analyze_data_quality(df_cleaned)
    
    # Check for required columns
    required_columns = ['vehicle_id', 'soh_capacity']
    missing_required = [col for col in required_columns if col not in df_cleaned.columns]
    
    if missing_required:
        print(f"\n⚠️ WARNING: Missing required columns: {missing_required}")
        print("Please check your cleaned data.")
        return None
    
    # Configuration for data preparation
    config = {
        'sequence_length': 180 if not DEBUG_MODE else 60,  # 30 min at 10s intervals (60 for debug)
        'batch_size': 32 if not DEBUG_MODE else 16,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'num_workers': 0  # Set to 0 for stability, increase for production
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if DEBUG_MODE:
        print("\n⚠️ DEBUG MODE ACTIVE: Using reduced parameters")
    
    # Prepare dataloaders
    try:
        print("\n" + "="*60)
        print("PREPARING DATALOADERS")
        print("="*60)
        
        prep_start = time.time()
        dataloaders = prepare_dataloaders(df_cleaned, config)
        prep_time = time.time() - prep_start
        
        print(f"\nData preparation completed in {prep_time:.2f}s")
        
        # Visualize samples
        visualize_sequences(dataloaders)
        
        # Save data configuration
        data_config = {
          'feature_names': dataloaders['feature_names'],
          'sequence_length': config['sequence_length'],
          'num_features': len(dataloaders['feature_names']),
          'target_column': 'soh_capacity',
          'vehicle_splits': dataloaders['vehicle_splits'],
          'prepared_data_dir': "prepared_data",
          'debug_mode': DEBUG_MODE
                      }

        
        joblib.dump(data_config, 'data_config.pkl')
        print(f"\nData configuration saved to data_config.pkl")
        
        # Final timing
        total_time = time.time() - total_start_time
        print(f"\n" + "="*60)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Training sequences: {len(dataloaders['train_loader'].dataset):,}")
        print(f"Features: {len(dataloaders['feature_names'])}")
        
        return dataloaders, data_config
        
    except Exception as e:
        print(f"\n❌ ERROR during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def save_prepared_arrays(dataloaders, output_dir="prepared_data"):
    """
    Save prepared sequences and targets to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    splits = {
        "train": dataloaders["train_dataset"],
        "val": dataloaders["val_dataset"],
        "test": dataloaders["test_dataset"]
    }

    for split, dataset in splits.items():
        X = dataset.sequences.astype(np.float32)
        y = dataset.targets.astype(np.float32)

        np.save(output_dir / f"X_{split}.npy", X)
        np.save(output_dir / f"y_{split}.npy", y)

        print(f"Saved {split}: X{X.shape}, y{y.shape}")

    print(f"\nPrepared arrays saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("STARTING DATA PREPARATION PIPELINE")
        print(f"Debug Mode: {DEBUG_MODE}")
        print("="*80)
        
        # Set environment variables for better performance
        import os
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        
        dataloaders, data_config = main_data_preparation()
        save_prepared_arrays(dataloaders, output_dir="prepared_data")
        if dataloaders:
            print("\n Phase 1 completed successfully!")
        else:
            print("\n Phase 1 failed!")
            
    except KeyboardInterrupt:
        print("\n\n Process interrupted by user")
    except Exception as e:
        print(f"\n Critical error in Phase 1: {e}")
        import traceback
        traceback.print_exc()