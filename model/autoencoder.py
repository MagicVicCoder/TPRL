import torch
import torch.nn as nn

class TokenAutoencoder(nn.Module):
    """
    Token-wise autoencoder for compressing visual token features.
    Maps each token from d_v dimensions to d_l dimensions (d_l << d_v).
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: d_v -> hidden -> d_l
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: d_l -> hidden -> d_v
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        Encode visual tokens to latent space.
        Args:
            x: [batch_size, num_tokens, input_dim]
        Returns:
            z: [batch_size, num_tokens, latent_dim]
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent codes back to original space.
        Args:
            z: [batch_size, num_tokens, latent_dim]
        Returns:
            x_recon: [batch_size, num_tokens, input_dim]
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full autoencoder forward pass.
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def reconstruction_loss(self, x, x_recon):
        """
        Compute MSE reconstruction loss.
        """
        return nn.functional.mse_loss(x_recon, x)
