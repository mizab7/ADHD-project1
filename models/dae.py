import torch
import torch.nn as nn
import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    """
    1D Denoising Autoencoder for fMRI ROI signals (116 AAL regions).
    Learns a bottleneck latent representation while removing high-frequency noise & artifacts.
    """
    def __init__(self, input_dim=116, latent_dim=32, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def add_noise(self, x):
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

    def forward(self, x):
        # Input shape: (Batch * Timepoints, 116)
        noisy_x = self.add_noise(x)
        latent = self.encoder(noisy_x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def denoise(self, x):
        """
        Denoises input tensor x of shape (B, N) or (B, T, N).
        """
        self.eval()
        with torch.no_grad():
            orig_shape = x.shape
            if len(orig_shape) == 3:
                B, T, N = orig_shape
                flat_x = x.reshape(-1, N)
                latent = self.encoder(flat_x)
                reconstructed = self.decoder(latent)
                return reconstructed.reshape(B, T, N)
            else:
                latent = self.encoder(x)
                return self.decoder(latent)

def train_dae_model(dae_model, data_tensor, epochs=30, batch_size=64, lr=1e-3, device="cpu"):
    """
    Trains DAE model on training fold data_tensor shape (N_samples, 116).
    """
    dae_model = dae_model.to(device)
    optimizer = optim.Adam(dae_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(data_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dae_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, _ = dae_model(x_batch)
            loss = criterion(reconstructed, x_batch)  # target is clean input x_batch
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = total_loss / len(data_tensor)
        if (epoch + 1) % 10 == 0:
            print(f"  [DAE Epoch {epoch+1}/{epochs}] MSE Loss: {epoch_loss:.6f}")
            
    dae_model.eval()
    return dae_model

if __name__ == "__main__":
    dummy_input = torch.randn(256, 116)
    dae = DenoisingAutoencoder(input_dim=116, latent_dim=32)
    rec, lat = dae(dummy_input)
    print(f"DAE Input shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {rec.shape}, Latent shape: {lat.shape}")
    denoised = dae.denoise(dummy_input)
    print(f"Denoised output shape: {denoised.shape}")
