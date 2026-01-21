"""
MODEL ARCHITECTURE FOR SOH ESTIMATION
Contains the neural network model definitions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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