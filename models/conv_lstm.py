import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DSpatialEncoder(nn.Module):
    """
    2D Convolutional Spatial Feature Extractor for 116 x 116 Functional Connectivity Matrices.
    Extracts spatial topological interaction features from brain ROI connectivity networks.
    """
    def __init__(self, in_channels=1, conv_channels=[32, 64], kernel_size=(3, 3)):
        super(Conv2DSpatialEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # 116x116 -> 58x58
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # 58x58 -> 29x29
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4) # 29x29 -> 7x7
        
        self.feature_dim = conv_channels[1] * 7 * 7
        
        # Placeholders for Grad-CAM explainability hooks
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Input shape: (Batch * Timepoints, 1, 116, 116)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        self.activations = x
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.pool3(x)
        
        flat_x = x.view(x.size(0), -1)
        return flat_x

class ConvLSTMClassifier(nn.Module):
    """
    End-to-End Spatiotemporal ConvLSTM Architecture for ADHD Classification.
    Input Shape: (Batch, T_windows, 1, 116, 116)
    Output: Logits (Batch, 2) or Probability (Batch, 2)
    """
    def __init__(self, in_channels=1, conv_channels=[32, 64], lstm_hidden_dim=128, num_classes=2, dropout=0.3):
        super(ConvLSTMClassifier, self).__init__()
        
        self.spatial_encoder = Conv2DSpatialEncoder(in_channels=in_channels, conv_channels=conv_channels)
        spatial_dim = self.spatial_encoder.feature_dim
        
        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T_windows, 1, 116, 116) or (B, T_windows, 116, 116)
        if x.ndim == 4:
            x = x.unsqueeze(2)  # Add channel dim: (B, T_windows, 1, 116, 116)
            
        B, T, C, H, W = x.shape
        
        # Flatten temporal and batch dimensions for 2D spatial convolution
        x_flat = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(x_flat)  # (B * T, spatial_dim)
        
        # Reshape back to sequence: (B, T, spatial_dim)
        sequence_features = spatial_features.view(B, T, -1)
        
        # Pass sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence_features)
        
        # Use final LSTM state for classification
        last_state = lstm_out[:, -1, :]  # (B, lstm_hidden_dim)
        
        logits = self.classifier(last_state)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

if __name__ == "__main__":
    # Test model with dummy input batch
    dummy_input = torch.randn(4, 10, 1, 116, 116)  # Batch=4, T_windows=10, Channels=1, 116x116
    model = ConvLSTMClassifier(conv_channels=[32, 64], lstm_hidden_dim=128)
    logits = model(dummy_input)
    proba = model.predict_proba(dummy_input)
    print(f"ConvLSTM Input shape: {dummy_input.shape}")
    print(f"Output Logits shape: {logits.shape}")
    print(f"Sample ADHD/Control probabilities:\n{proba}")
