"""
OPTIMIZED DEEP LEARNING MODEL FOR SOH ESTIMATION
Simplified architecture for faster training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import joblib

class SimplifiedSOHModel(nn.Module):
    """
    Simplified hybrid model for SOH estimation
    Combines LSTM and CNN for efficiency
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config['num_features']
        self.sequence_length = config.get('sequence_length', 180)
        
        # 1. LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # 2. 1D CNN for local patterns
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(self.input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Calculate CNN output size
        cnn_out_length = self.sequence_length // 2  # After maxpool
        cnn_out_features = 64 * cnn_out_length
        
        # LSTM output size (bidirectional)
        lstm_out_features = 64 * 2  # 64 hidden * 2 directions
        
        # Combined features
        total_features = lstm_out_features + cnn_out_features
        
        # 3. Attention layer for important timesteps
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_features, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 4. Fusion and regression layers
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # 5. Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # 6. Uncertainty estimation (simplified)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensure positive
        )
        
        print(f"Model initialized:")
        print(f"  Input: {self.input_dim} features × {self.sequence_length} timesteps")
        print(f"  LSTM output: {lstm_out_features} features")
        print(f"  CNN output: {cnn_out_features} features")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        x shape: (batch, seq_len, features)
        """
        batch_size = x.shape[0]
        
        # === LSTM Branch ===
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 128)
        
        # Attention over timesteps
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        lstm_attended = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, 128)
        
        # === CNN Branch ===
        # Reshape for CNN: (batch, features, seq_len)
        cnn_input = x.transpose(1, 2)  # (batch, features, seq_len)
        cnn_out = self.cnn(cnn_input)  # (batch, 64, seq_len/2)
        
        # Flatten CNN output
        cnn_flat = cnn_out.view(batch_size, -1)  # (batch, 64 * (seq_len/2))
        
        # === Fusion ===
        combined = torch.cat([lstm_attended, cnn_flat], dim=1)  # (batch, total_features)
        fused = self.fusion(combined)
        
        # === Outputs ===
        prediction = self.regression_head(fused)
        uncertainty = self.uncertainty_head(fused)
        
        return prediction, uncertainty, attn_weights.squeeze(-1)


class EfficientLoss(nn.Module):
    """Efficient loss function for SOH estimation"""
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, predictions, targets, uncertainties=None):
        """
        Calculate combined loss
        """
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # MAE for robustness
        mae_loss = self.mae(predictions, targets)
        
        # Combined base loss
        base_loss = 0.7 * mse_loss + 0.3 * mae_loss
        
        # Uncertainty calibration (optional)
        if uncertainties is not None:
            # Simple uncertainty penalty
            uncertainty_loss = uncertainties.mean()
            base_loss = base_loss + 0.05 * uncertainty_loss
        
        return base_loss


class EfficientTrainer:
    """Efficient trainer for SOH model"""
    
    def __init__(self, config, device=None):
        self.config = config
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SimplifiedSOHModel(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = EfficientLoss(alpha=config.get('alpha', 0.1))
        
        # History tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_rmse': [], 'val_rmse': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch in pbar:
            # Move to device
            sequences = batch['sequence'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, uncertainty, _ = self.model(sequences)
            
            # Calculate loss
            loss = self.criterion(predictions, targets, uncertainty)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().numpy().flatten())
            all_targets.extend(targets.detach().cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        
        # Calculate epoch metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, mae, rmse
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                predictions, uncertainty, _ = self.model(sequences)
                
                # Calculate loss
                loss = self.criterion(predictions, targets, uncertainty)
                total_loss += loss.item()
                
                # Store predictions
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_uncertainties.extend(uncertainty.cpu().numpy().flatten())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_uncertainties = np.array(all_uncertainties)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        avg_loss = total_loss / len(val_loader)
        
        # Calculate R²
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate coverage
        lower = all_preds - 1.96 * all_uncertainties
        upper = all_preds + 1.96 * all_uncertainties
        coverage = np.mean((all_targets >= lower) & (all_targets <= upper))
        
        return avg_loss, mae, rmse, r2, coverage, all_preds, all_targets, all_uncertainties
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=15):
        """Main training loop"""
        
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        
        no_improve = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_mae, val_rmse, val_r2, val_coverage, _, _, _ = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"\nEpoch {epoch:3d}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.3f}%, RMSE: {train_rmse:.3f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.3f}%, RMSE: {val_rmse:.3f}%")
            print(f"         R²: {val_r2:.3f}, Coverage: {val_coverage:.3f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                no_improve = 0
                print(f"  ↳ New best model! (loss: {val_loss:.4f})")
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'history': self.history
                }, 'best_model.pth')
            else:
                no_improve += 1
                print(f"  ↳ No improvement for {no_improve} epoch(s)")
            
            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model (val_loss: {self.best_val_loss:.4f})")
        
        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        
        test_loss, test_mae, test_rmse, test_r2, test_coverage, preds, targets, uncertainties = self.validate(test_loader)
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"MAE:  {test_mae:.3f}%")
        print(f"RMSE: {test_rmse:.3f}%")
        print(f"R²:   {test_r2:.3f}")
        print(f"Coverage (95%): {test_coverage:.3f}")
        print(f"Avg Uncertainty: {np.mean(uncertainties):.3f}")
        
        return {
            'predictions': preds,
            'targets': targets,
            'uncertainties': uncertainties,
            'metrics': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'coverage': test_coverage
            }
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(epochs, self.history['train_mae'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_mae'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (%)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE
        axes[1, 0].plot(epochs, self.history['train_rmse'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_rmse'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE (%)')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history_simple.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, test_results):
        """Plot predictions vs actual"""
        preds = test_results['predictions']
        targets = test_results['targets']
        uncertainties = test_results['uncertainties']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(targets, preds, alpha=0.5, s=20)
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual SOH (%)')
        axes[0].set_ylabel('Predicted SOH (%)')
        axes[0].set_title('Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add error bars for uncertainty
        axes[0].errorbar(targets[::10], preds[::10], 
                        yerr=1.96 * uncertainties[::10],
                        fmt='o', alpha=0.5, capsize=3)
        
        # Residuals
        residuals = preds - targets
        axes[1].scatter(preds, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted SOH (%)')
        axes[1].set_ylabel('Residual (%)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Add histogram of residuals
        ax_hist = axes[1].inset_axes([0.6, 0.6, 0.35, 0.35])
        ax_hist.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax_hist.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax_hist.set_xlabel('Residual')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Residual Distribution')
        
        plt.suptitle('Model Predictions Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('predictions_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_final_model(self, path='final_model_simple.pth'):
        """Save final model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"\nFinal model saved to {path}")


def quick_model_test(dataloaders, config):
    """Quick test to ensure everything works"""
    print("Running quick model test...")
    
    # Get one batch
    batch = next(iter(dataloaders['train_loader']))
    
    print(f"Batch shape: {batch['sequence'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    
    # Initialize model
    model = SimplifiedSOHModel(config)
    
    # Test forward pass
    with torch.no_grad():
        preds, uncert, attn = model(batch['sequence'])
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Uncertainty shape: {uncert.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Test loss
    criterion = EfficientLoss()
    loss = criterion(preds, batch['target'], uncert)
    print(f"Loss value: {loss.item():.4f}")
    
    print("✅ Quick test passed!")
    return True


def main_training_pipeline(dataloaders, data_config):
    """Main training pipeline"""
    
    print("="*80)
    print("PHASE 2: SIMPLIFIED MODEL TRAINING")
    print("="*80)
    
    # Model configuration
    model_config = {
        **data_config,  # Includes num_features, sequence_length
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'alpha': 0.1
    }
    
    # Quick test
    quick_model_test(dataloaders, model_config)
    
    # Initialize trainer
    trainer = EfficientTrainer(model_config)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        num_epochs=50,
        patience=15
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_results = trainer.evaluate(dataloaders['test_loader'])
    
    # Plot predictions
    trainer.plot_predictions(test_results)
    
    # Save final model
    trainer.save_final_model()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Test MAE:  {test_results['metrics']['mae']:.3f}%")
    print(f"Test RMSE: {test_results['metrics']['rmse']:.3f}%")
    print(f"Test R²:   {test_results['metrics']['r2']:.3f}")
    
    return trainer, test_results


def run_standalone():
    """Run standalone if dataloaders not available"""
    print("Standalone mode - Loading saved configuration...")
    
    try:
        # Load data configuration
        data_config = joblib.load('data_config.pkl')
        
        print(f"Loaded configuration:")
        print(f"  Features: {data_config['num_features']}")
        print(f"  Sequence length: {data_config['sequence_length']}")
        
        # Create dummy dataloaders for testing
        class DummyDataset:
            def __init__(self):
                self.sequences = torch.randn(100, data_config['sequence_length'], data_config['num_features'])
                self.targets = torch.randn(100, 1) * 10 + 80  # SOH around 80-90%
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return {
                    'sequence': self.sequences[idx],
                    'target': self.targets[idx]
                }
        
        from torch.utils.data import DataLoader
        
        dummy_dataset = DummyDataset()
        dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        
        dataloaders = {
            'train_loader': dummy_loader,
            'val_loader': dummy_loader,
            'test_loader': dummy_loader
        }
        
        # Run training
        trainer, results = main_training_pipeline(dataloaders, data_config)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease run Phase 1 first to create proper dataloaders.")


if __name__ == "__main__":
    # Check if we have CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Run training
    print("\nTo run properly, execute from main script after Phase 1")
    print("For testing, run: run_standalone()")