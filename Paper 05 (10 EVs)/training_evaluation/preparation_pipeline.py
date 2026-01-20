"""
DEEP LEARNING DATA PREPARATION FOR SOH ESTIMATION
Phase 1: Sequence Creation and Feature Engineering
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

class SOHSequenceDataset(Dataset):
    """Create sequence datasets for deep learning models"""
    
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
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.step_size = step_size
        self.mode = mode
        
        # Filter by vehicle if specified
        if vehicle_ids is not None:
            self.df = df[df['vehicle_id'].isin(vehicle_ids)].copy()
        else:
            self.df = df.copy()
        
        # Define feature groups
        self._define_feature_groups()
        
        # Prepare sequences
        self.sequences, self.targets, self.metadata = self._create_sequences()
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        print(f"Features: {len(self.feature_cols)}")
    
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
        
        # Group 4: Encoded features will be handled separately
        
        # Combine available features
        self.feature_cols = []
        for feat_list in [self.raw_features, self.derived_features]:
            for feat in feat_list:
                if feat in self.df.columns:
                    self.feature_cols.append(feat)
        
        # Add rolling statistics
        self._add_rolling_features()
    
    def _add_rolling_features(self):
        """Add rolling window statistics as features"""
        
        print("Creating rolling window features...")
        
        # Group by vehicle and session for proper rolling
        for vehicle_id in self.df['vehicle_id'].unique():
            vehicle_mask = self.df['vehicle_id'] == vehicle_id
            
            for session in self.df.loc[vehicle_mask, 'session_change'].unique():
                session_mask = vehicle_mask & (self.df['session_change'] == session)
                
                # Skip short sessions
                if session_mask.sum() < 30:
                    continue
                
                # Session data
                session_data = self.df.loc[session_mask].copy()
                
                # 5-minute rolling features (30 samples at 10s intervals)
                if 'hv_current' in session_data.columns:
                    self.df.loc[session_mask, 'rolling_mean_current_5min'] = (
                        session_data['hv_current'].rolling(window=30, min_periods=5).mean()
                    )
                    self.df.loc[session_mask, 'rolling_std_current_5min'] = (
                        session_data['hv_current'].rolling(window=30, min_periods=5).std()
                    )
                
                if 'bcell_maxVoltage' in session_data.columns:
                    self.df.loc[session_mask, 'rolling_voltage_range_5min'] = (
                        session_data['bcell_maxVoltage'].rolling(window=30, min_periods=5).max() -
                        session_data['bcell_minVoltage'].rolling(window=30, min_periods=5).min()
                    )
                
                if 'bcell_maxTemp' in session_data.columns:
                    self.df.loc[session_mask, 'temp_variance_5min'] = (
                        session_data['bcell_maxTemp'].rolling(window=30, min_periods=5).var()
                    )
        
        # Add cumulative features
        if 'hv_current' in self.df.columns:
            self.df['cumulative_charge'] = np.where(
                self.df['hv_current'] > 0, 
                self.df['hv_current'].abs(), 
                0
            ).groupby(self.df['session_change']).cumsum()
            
            self.df['cumulative_discharge'] = np.where(
                self.df['hv_current'] < 0, 
                self.df['hv_current'].abs(), 
                0
            ).groupby(self.df['session_change']).cumsum()
        
        # Add statistical features to list
        self.stat_features = [
            'rolling_mean_current_5min', 'rolling_std_current_5min',
            'rolling_voltage_range_5min', 'temp_variance_5min',
            'cumulative_charge', 'cumulative_discharge'
        ]
        
        # Add to feature columns
        for feat in self.stat_features:
            if feat in self.df.columns:
                self.feature_cols.append(feat)
        
        # Fill NaN values in new features
        self.df[self.stat_features] = self.df[self.stat_features].fillna(method='ffill').fillna(0)
    
    def _create_sequences(self):
        """Create overlapping sequences from time series data"""
        
        sequences = []
        targets = []
        metadata = []  # Store vehicle_id, session_id, timestamp
        
        # Group by vehicle and session
        grouped = self.df.groupby(['vehicle_id', 'session_change'])
        
        for (vehicle_id, session_id), group in tqdm(grouped, desc="Creating sequences"):
            # Sort by time
            group = group.sort_values('relative_time')
            
            # Ensure we have target values
            if self.target_col not in group.columns or group[self.target_col].isna().all():
                continue
            
            # Get feature matrix
            feature_data = group[self.feature_cols].values
            target_data = group[self.target_col].values
            
            # Create sequences with sliding window
            for i in range(0, len(feature_data) - self.sequence_length, self.step_size):
                seq_end = i + self.sequence_length
                
                # Check if we have valid target at the end of sequence
                if not np.isnan(target_data[seq_end - 1]):
                    sequence = feature_data[i:seq_end]
                    target = target_data[seq_end - 1]  # Predict SOH at end of sequence
                    
                    # Only include if sequence doesn't have too many NaNs
                    if np.isnan(sequence).mean() < 0.1:  # Less than 10% NaN
                        sequences.append(sequence)
                        targets.append(target)
                        metadata.append({
                            'vehicle_id': vehicle_id,
                            'session_id': session_id,
                            'start_idx': i,
                            'end_idx': seq_end
                        })
        
        # Convert to numpy arrays
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        # Handle NaN values in sequences
        for i in range(len(sequences_array)):
            seq = sequences_array[i]
            # Forward fill NaN within sequence
            df_seq = pd.DataFrame(seq)
            df_seq = df_seq.fillna(method='ffill').fillna(method='bfill').fillna(0)
            sequences_array[i] = df_seq.values
        
        return sequences_array, targets_array, metadata
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        return {
            'sequence': sequence,
            'target': target,
            'metadata': self.metadata[idx]
        }
    
    def get_feature_names(self):
        return self.feature_cols


class DataScaler:
    """Handle scaling and normalization of sequences"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scalers = {}  # One scaler per feature
        
    def fit(self, sequences):
        """Fit scalers on training data"""
        # sequences shape: (n_samples, seq_len, n_features)
        n_features = sequences.shape[2]
        
        for feature_idx in range(n_features):
            # Extract all values for this feature across all sequences
            feature_values = sequences[:, :, feature_idx].reshape(-1, 1)
            
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
            
            # Fit on non-NaN values
            non_nan_mask = ~np.isnan(feature_values.flatten())
            if non_nan_mask.sum() > 0:
                scaler.fit(feature_values[non_nan_mask])
            
            self.scalers[feature_idx] = scaler
        
        return self
    
    def transform(self, sequences):
        """Transform sequences using fitted scalers"""
        transformed = sequences.copy()
        n_samples, seq_len, n_features = sequences.shape
        
        for feature_idx in range(n_features):
            scaler = self.scalers.get(feature_idx)
            if scaler:
                # Reshape, transform, reshape back
                feature_values = sequences[:, :, feature_idx].reshape(-1, 1)
                transformed_values = scaler.transform(feature_values)
                transformed[:, :, feature_idx] = transformed_values.reshape(n_samples, seq_len)
        
        return transformed
    
    def save(self, path):
        """Save scalers to disk"""
        joblib.dump(self.scalers, path)
    
    def load(self, path):
        """Load scalers from disk"""
        self.scalers = joblib.load(path)
        return self


def prepare_dataloaders(df, config):
    """
    Prepare train/val/test dataloaders with proper splitting
    """
    
    # Configuration
    seq_length = config.get('sequence_length', 360)
    batch_size = config.get('batch_size', 32)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    random_seed = config.get('random_seed', 42)
    
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
    
    print(f"Train vehicles: {train_vehicles}")
    print(f"Val vehicles: {val_vehicles}")
    print(f"Test vehicles: {test_vehicles}")
    
    # Create datasets
    train_dataset = SOHSequenceDataset(
        df[df['vehicle_id'].isin(train_vehicles)],
        sequence_length=seq_length,
        mode='train'
    )
    
    val_dataset = SOHSequenceDataset(
        df[df['vehicle_id'].isin(val_vehicles)],
        sequence_length=seq_length,
        mode='val'
    )
    
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
    train_dataset.sequences = scaler.transform(train_dataset.sequences)
    val_dataset.sequences = scaler.transform(val_dataset.sequences)
    test_dataset.sequences = scaler.transform(test_dataset.sequences)
    
    # Save scaler
    scaler.save('data_scaler.pkl')
    print(f"Scaler saved to data_scaler.pkl")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 2)
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
    
    # Show feature names
    print("\nFeatures used:")
    for i, feat in enumerate(train_dataset.get_feature_names()):
        print(f"  {i:2d}. {feat}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'scaler': scaler,
        'feature_names': train_dataset.get_feature_names(),
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
    batch = next(iter(train_loader))
    sequences = batch['sequence']
    targets = batch['target']
    
    print(f"Batch shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Plot sample sequences
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Select random features to plot
    n_features = sequences.shape[2]
    feature_indices = np.random.choice(n_features, min(8, n_features), replace=False)
    
    for idx, (ax, feat_idx) in enumerate(zip(axes, feature_indices)):
        # Plot first few samples
        for sample_idx in range(min(3, sequences.shape[0])):
            ax.plot(sequences[sample_idx, :, feat_idx].numpy(), 
                   alpha=0.7, linewidth=1)
        
        ax.set_title(f"Feature: {feature_names[feat_idx]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)
        ax.legend([f"Sample {i}" for i in range(min(3, sequences.shape[0]))])
    
    plt.suptitle("Sample Sequences (Multiple Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sequence_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot target distribution
    all_targets = []
    for batch in train_loader:
        all_targets.extend(batch['target'].numpy().flatten())
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_targets, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('SOH (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of SOH Targets in Training Set')
    plt.grid(True, alpha=0.3)
    plt.savefig('soh_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def main_data_preparation():
    """Main data preparation pipeline"""
    
    print("="*80)
    print("PHASE 1: DEEP LEARNING DATA PREPARATION")
    print("="*80)
    
    # Load cleaned data
    data_path = Path(r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs")
    df_cleaned = pd.read_parquet(data_path / "all_vehicles_cleaned.parquet")
    
    print(f"Loaded cleaned data: {df_cleaned.shape}")
    print(f"Columns: {list(df_cleaned.columns)}")
    
    # Configuration for data preparation
    config = {
        'sequence_length': 360,  # 60 minutes at 10s intervals
        'batch_size': 32,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'num_workers': 4
    }
    
    # Prepare dataloaders
    dataloaders = prepare_dataloaders(df_cleaned, config)
    
    # Visualize samples
    visualize_sequences(dataloaders)
    
    # Save data configuration
    data_config = {
        'feature_names': dataloaders['feature_names'],
        'sequence_length': config['sequence_length'],
        'vehicle_splits': dataloaders['vehicle_splits'],
        'num_features': len(dataloaders['feature_names'])
    }
    
    joblib.dump(data_config, 'data_config.pkl')
    print(f"\nData configuration saved to data_config.pkl")
    
    return dataloaders, data_config


if __name__ == "__main__":
    dataloaders, data_config = main_data_preparation()