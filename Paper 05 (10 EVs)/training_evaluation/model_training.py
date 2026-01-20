"""
PHASE 2: DEEP LEARNING MODEL ARCHITECTURES
Hybrid models for SOH estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalAttention(nn.Module):
    """Attention mechanism for temporal sequences"""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x + attn_output)
        return output, attn_weights


class TemporalBranch(nn.Module):
    """LSTM-based temporal dynamics branch"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Global average pooling over time dimension
        pooled = attn_out.mean(dim=1)  # (batch, hidden_dim*2)
        
        return pooled, attn_weights


class SpatialBranch(nn.Module):
    """CNN-based spatial pattern extraction branch"""
    
    def __init__(self, input_dim, cnn_channels=[64, 128], kernel_sizes=[5, 3], dropout=0.2):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Treat features as channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Reshape for CNN: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        # Treat input_dim as channels
        x = x.unsqueeze(1)  # (batch, 1, input_dim, seq_len)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global pooling over sequence length
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        
        return x


class TransformerBranch(nn.Module):
    """Transformer-based branch for capturing long-range dependencies"""
    
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Linear projection to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch, d_model) for positional encoding compatibility
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.output_proj(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SOHHybridModel(nn.Module):
    """Hybrid model combining LSTM, CNN, and Transformer branches"""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract configuration
        self.input_dim = config['num_features']
        self.sequence_length = config.get('sequence_length', 360)
        
        # Branch configurations
        temporal_hidden = config.get('temporal_hidden', 128)
        cnn_channels = config.get('cnn_channels', [64, 128])
        d_model = config.get('d_model', 128)
        
        # Initialize branches
        self.temporal_branch = TemporalBranch(
            input_dim=self.input_dim,
            hidden_dim=temporal_hidden,
            num_layers=config.get('temporal_layers', 2),
            dropout=config.get('temporal_dropout', 0.3)
        )
        
        self.spatial_branch = SpatialBranch(
            input_dim=self.input_dim,
            cnn_channels=cnn_channels,
            kernel_sizes=config.get('kernel_sizes', [5, 3]),
            dropout=config.get('spatial_dropout', 0.2)
        )
        
        self.transformer_branch = TransformerBranch(
            input_dim=self.input_dim,
            d_model=d_model,
            nhead=config.get('nhead', 4),
            num_layers=config.get('transformer_layers', 2),
            dropout=config.get('transformer_dropout', 0.1)
        )
        
        # Calculate combined feature dimensions
        temporal_out = temporal_hidden * 2  # bidirectional
        spatial_out = cnn_channels[-1] if cnn_channels else 128
        
        combined_features = temporal_out + spatial_out + d_model
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.get('fusion_dropout', 0.3)),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.get('fusion_dropout', 0.3)),
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x, return_attention=False):
        """
        Forward pass through hybrid model
        x: (batch, seq_len, input_dim)
        Returns: predictions, uncertainty, (optional: attention weights)
        """
        # Extract features from each branch
        temporal_features, attn_weights = self.temporal_branch(x)
        spatial_features = self.spatial_branch(x)
        transformer_features = self.transformer_branch(x)
        
        # Concatenate branch outputs
        combined = torch.cat([
            temporal_features,
            spatial_features,
            transformer_features
        ], dim=-1)
        
        # Fusion layers
        fused = self.fusion(combined)
        
        # Regression prediction
        soh_pred = self.regression_head(fused)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(fused)
        
        if return_attention:
            return soh_pred, uncertainty, attn_weights
        else:
            return soh_pred, uncertainty


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function for SOH estimation"""
    
    def __init__(self, alpha=0.1, beta=0.05, gamma=0.01):
        super().__init__()
        self.alpha = alpha  # Weight for monotonicity loss
        self.beta = beta    # Weight for smoothness loss
        self.gamma = gamma  # Weight for physics loss
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predictions, targets, uncertainties=None, previous_preds=None):
        """
        Calculate combined loss with physics constraints
        
        Args:
            predictions: Current batch predictions
            targets: Ground truth
            uncertainties: Predicted uncertainties
            previous_preds: Previous predictions for monotonicity check
        """
        # Base regression loss
        base_loss = self.mse_loss(predictions, targets)
        
        # MAE for robustness
        mae_loss = self.mae_loss(predictions, targets)
        
        total_loss = base_loss + 0.5 * mae_loss
        
        # Uncertainty calibration loss (if uncertainties provided)
        if uncertainties is not None:
            # Negative log likelihood for proper uncertainty calibration
            nll_loss = 0.5 * torch.mean(torch.log(uncertainties**2) + 
                                        (predictions - targets)**2 / (uncertainties**2 + 1e-8))
            total_loss = total_loss + 0.1 * nll_loss
        
        # Monotonicity constraint: SOH should not increase over time
        if previous_preds is not None and len(previous_preds) > 0:
            monotonic_loss = F.relu(predictions - previous_preds).mean()
            total_loss = total_loss + self.alpha * monotonic_loss
        
        return total_loss


class SOHModelManager:
    """Manager class for handling model training and evaluation"""
    
    def __init__(self, config, device=None):
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SOHHybridModel(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = PhysicsInformedLoss(
            alpha=config.get('alpha', 0.1),
            beta=config.get('beta', 0.05),
            gamma=config.get('gamma', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rates': []
        }
        
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different parts"""
        
        # Separate parameters
        temporal_params = list(self.model.temporal_branch.parameters())
        spatial_params = list(self.model.spatial_branch.parameters())
        transformer_params = list(self.model.transformer_branch.parameters())
        fusion_params = list(self.model.fusion.parameters())
        head_params = list(self.model.regression_head.parameters()) + \
                     list(self.model.uncertainty_head.parameters())
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': temporal_params, 'lr': self.config.get('lr_temporal', 1e-3)},
            {'params': spatial_params, 'lr': self.config.get('lr_spatial', 1e-3)},
            {'params': transformer_params, 'lr': self.config.get('lr_transformer', 1e-3)},
            {'params': fusion_params, 'lr': self.config.get('lr_fusion', 1e-3)},
            {'params': head_params, 'lr': self.config.get('lr_head', 1e-3)},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        return optimizer
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            sequences = batch['sequence'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            soh_pred, uncertainty = self.model(sequences)
            
            # Calculate loss
            loss = self.criterion(soh_pred, target, uncertainty)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions.extend(soh_pred.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, mae, rmse
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                soh_pred, uncertainty = self.model(sequences)
                
                # Calculate loss
                loss = self.criterion(soh_pred, target, uncertainty)
                total_loss += loss.item()
                
                # Store predictions
                predictions.extend(soh_pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                uncertainties.extend(uncertainty.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        uncertainties = np.array(uncertainties).flatten()
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        avg_loss = total_loss / len(val_loader)
        
        # Calculate coverage of uncertainty intervals
        coverage_95 = self._calculate_coverage(predictions, uncertainties, targets)
        
        return avg_loss, mae, rmse, coverage_95, predictions, targets, uncertainties
    
    def _calculate_coverage(self, predictions, uncertainties, targets, confidence=0.95):
        """Calculate coverage of uncertainty intervals"""
        z_score = 1.96  # For 95% confidence
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
        return coverage
    
    def train(self, train_loader, val_loader, num_epochs=100, patience=20):
        """Full training loop"""
        
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_mae, val_rmse, coverage, _, _, _ = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print progress
            print(f"\nEpoch {epoch:3d}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f}%, RMSE: {train_rmse:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, MAE: {val_mae:.2f}%, RMSE: {val_rmse:.2f}%")
            print(f"  Uncertainty Coverage (95%): {coverage:.3f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
                print(f"  ↳ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ↳ No improvement for {patience_counter} epoch(s)")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Load best model
        self.load_model('best_model.pth')
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def save_model(self, path):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        
        val_loss, mae, rmse, coverage, predictions, targets, uncertainties = self.validate(test_loader)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        print(f"MAE:  {mae:.3f}%")
        print(f"RMSE: {rmse:.3f}%")
        print(f"R² Score: {self._calculate_r2(predictions, targets):.3f}")
        print(f"Uncertainty Coverage (95%): {coverage:.3f}")
        print(f"Average Uncertainty: {np.mean(uncertainties):.3f}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'coverage': coverage
            }
        }
    
    def _calculate_r2(self, predictions, targets):
        """Calculate R² score"""
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def visualize_training(self):
        """Visualize training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot training and validation loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train')
        axes[0, 1].plot(self.history['val_mae'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (%)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot RMSE
        axes[0, 2].plot(self.history['train_rmse'], label='Train')
        axes[0, 2].plot(self.history['val_rmse'], label='Validation')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE (%)')
        axes[0, 2].set_title('Root Mean Square Error')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot loss vs MAE
        axes[1, 1].scatter(self.history['train_loss'], self.history['train_mae'], 
                          alpha=0.5, label='Train', s=10)
        axes[1, 1].scatter(self.history['val_loss'], self.history['val_mae'], 
                          alpha=0.5, label='Validation', s=10)
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('MAE (%)')
        axes[1, 1].set_title('Loss vs MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.suptitle('Training History Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


def main_model_training(dataloaders, data_config):
    """Main function for model training"""
    
    print("="*80)
    print("PHASE 2: DEEP LEARNING MODEL TRAINING")
    print("="*80)
    
    # Update data_config with model parameters
    model_config = {
        **data_config,  # Include sequence_length, num_features
        'temporal_hidden': 128,
        'temporal_layers': 2,
        'temporal_dropout': 0.3,
        'cnn_channels': [64, 128],
        'kernel_sizes': [5, 3],
        'spatial_dropout': 0.2,
        'd_model': 128,
        'nhead': 4,
        'transformer_layers': 2,
        'transformer_dropout': 0.1,
        'fusion_dropout': 0.3,
        'lr_temporal': 1e-3,
        'lr_spatial': 1e-3,
        'lr_transformer': 1e-3,
        'lr_fusion': 1e-3,
        'lr_head': 1e-3,
        'weight_decay': 1e-4,
        'alpha': 0.1,
        'beta': 0.05,
        'gamma': 0.01
    }
    
    # Initialize model manager
    model_manager = SOHModelManager(model_config)
    
    # Print model architecture
    print("\nModel Architecture:")
    print("="*40)
    print(f"Input dimension: {model_config['num_features']}")
    print(f"Sequence length: {model_config['sequence_length']}")
    print(f"Total parameters: {sum(p.numel() for p in model_manager.model.parameters()):,}")
    
    # Train model
    history = model_manager.train(
        dataloaders['train_loader'],
        dataloaders['val_loader'],
        num_epochs=100,
        patience=20
    )
    
    # Visualize training
    model_manager.visualize_training()
    
    # Evaluate on test set
    test_results = model_manager.evaluate(dataloaders['test_loader'])
    
    # Save final model
    model_manager.save_model('final_model.pth')
    print(f"\nFinal model saved to final_model.pth")
    
    return model_manager, test_results


if __name__ == "__main__":
    # Assuming dataloaders and data_config are available from Phase 1
    # For standalone testing, load them:
    import joblib
    data_config = joblib.load('data_config.pkl')
    # dataloaders would need to be recreated or saved
    
    print("Run Phase 1 first to create dataloaders, then run Phase 2")