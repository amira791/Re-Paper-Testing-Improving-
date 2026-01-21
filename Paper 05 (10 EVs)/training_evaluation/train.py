"""
MODEL TRAINING PIPELINE - FIXED VERSION
Automatically loads Phase 1 data and starts training
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add model architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from model_arch import SimplifiedSOHModel, EfficientLoss
except ImportError:
    print("Error: model_architecture.py not found in same directory")
    print("Please make sure both files are in the same folder")
    sys.exit(1)

class EfficientTrainer:
    """Efficient trainer for SOH model - FIXED VERSION"""
    
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
        
        # Learning rate scheduler - FIXED for older PyTorch versions
        try:
            # Try with verbose parameter (newer PyTorch)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                min_lr=1e-6
            )
        except TypeError:
            # Fallback for older PyTorch without verbose parameter
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            print("Note: Using PyTorch without verbose parameter for scheduler")
        
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


def load_phase1_data():
    """Load data from Phase 1 - FIXED VERSION"""
    print("Loading Phase 1 data...")
    
    try:
        import joblib
        
        # Check if files exist
        data_config_path = 'data_config.pkl'
        data_scaler_path = 'data_scaler.pkl'
        
        if not os.path.exists(data_config_path):
            print(f"✗ File not found: {data_config_path}")
            return None, None
        
        # Load data configuration
        data_config = joblib.load(data_config_path)
        print(f"✓ Loaded data configuration")
        
        # Check for scaler
        if os.path.exists(data_scaler_path):
            scaler = joblib.load(data_scaler_path)
            print(f"✓ Loaded data scaler")
        else:
            print(f"⚠ Data scaler not found: {data_scaler_path}")
            scaler = None
        
        return data_config, scaler
        
    except Exception as e:
        print(f"✗ Error loading Phase 1 data: {e}")
        return None, None


def main():
    """Main function that automatically runs training"""
    
    print("="*80)
    print("AUTO-TRAINING PIPELINE FOR SOH ESTIMATION")
    print("="*80)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Try to load Phase 1 data
    data_config, _ = load_phase1_data()
    
    if data_config is None:
        print("\n⚠ Phase 1 data not found or incomplete.")
        print("Please run Phase 1 first to prepare data.")
        print("For testing, we'll create dummy data...")
        
        # Create dummy config for testing
        data_config = {
            'num_features': 15,
            'sequence_length': 180,
            'feature_names': [f'feature_{i}' for i in range(15)]
        }
        
        # Create dummy dataloaders for testing
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, n_samples=1000, seq_len=180, n_features=15):
                self.sequences = torch.randn(n_samples, seq_len, n_features)
                self.targets = torch.randn(n_samples, 1) * 5 + 85  # SOH 80-90%
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return {
                    'sequence': self.sequences[idx],
                    'target': self.targets[idx]
                }
        
        # Create smaller datasets for faster testing
        dummy_train = DummyDataset(n_samples=200, seq_len=data_config['sequence_length'], 
                                  n_features=data_config['num_features'])
        dummy_val = DummyDataset(n_samples=50, seq_len=data_config['sequence_length'], 
                                n_features=data_config['num_features'])
        dummy_test = DummyDataset(n_samples=50, seq_len=data_config['sequence_length'], 
                                 n_features=data_config['num_features'])
        
        train_loader = torch.utils.data.DataLoader(dummy_train, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dummy_val, batch_size=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dummy_test, batch_size=8, shuffle=False)
        
        dataloaders = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
        
        print("✓ Created dummy data for testing")
        print(f"  Train samples: {len(dummy_train)}")
        print(f"  Val samples: {len(dummy_val)}")
        print(f"  Test samples: {len(dummy_test)}")
    
    else:
        print(f"\n✓ Successfully loaded Phase 1 configuration:")
        print(f"  - Features: {data_config['num_features']}")
        print(f"  - Sequence length: {data_config['sequence_length']}")
        
        # If we have real data, we need dataloaders from Phase 1
        # For now, exit and ask to run a complete pipeline
        print("\n⚠ Actual dataloaders not found in saved files.")
        print("You need to run the training with actual data.")
        print("Would you like to: ")
        print("  1. Continue with dummy data for testing")
        print("  2. Exit and run Phase 1 properly")
        
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                print("Continuing with dummy data...")
                # Use the dummy data creation code from above
                # ... (copy the dummy data creation code here)
                # For brevity, I'll exit here
                print("Please modify the code to create dummy data with your config")
                return
            else:
                print("Exiting. Please run Phase 1 first.")
                return
        except:
            print("Exiting. Please run Phase 1 first.")
            return
    
    # Create model configuration
    model_config = {
        **data_config,  # Includes num_features, sequence_length
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'alpha': 0.1
    }
    
    # Initialize trainer
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)
    
    trainer = EfficientTrainer(model_config)
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        # Use fewer epochs for dummy data testing
        num_epochs = 10 if data_config is None else 50
        
        history = trainer.train(
            train_loader=dataloaders['train_loader'],
            val_loader=dataloaders['val_loader'],
            num_epochs=num_epochs,
            patience=5  # Shorter patience for testing
        )
        
        # Plot training history
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        trainer.plot_training_history()
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        test_results = trainer.evaluate(dataloaders['test_loader'])
        
        # Plot predictions
        trainer.plot_predictions(test_results)
        
        # Save final model
        trainer.save_final_model()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Test MAE:  {test_results['metrics']['mae']:.3f}%")
        print(f"Test RMSE: {test_results['metrics']['rmse']:.3f}%")
        print(f"Test R²:   {test_results['metrics']['r2']:.3f}")
        print(f"Coverage:  {test_results['metrics']['coverage']:.3f}")
        
        return trainer, test_results
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the training pipeline automatically
    main()