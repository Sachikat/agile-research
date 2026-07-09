
"""
Revised moth yaw prediction model (Version B)

Key changes:
- Shared encoder across species
- Learned species embeddings
- Species-level tz normalization
- No balanced subsampling
- Latent dim = 8
- Stronger predictor network
- Lighter reconstruction loss
- Early stopping

NOTE:
This is a replacement architecture for the original model.
You can transplant your plotting/SHAP/export utilities from the original file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 8
SPECIES_EMBED_DIM = 4


class MotorProgramModel(nn.Module):

    def __init__(self, input_dim, num_species):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, LATENT_DIM),
        )

        self.species_embedding = nn.Embedding(
            num_species,
            SPECIES_EMBED_DIM
        )

        self.predictor = nn.Sequential(
            nn.Linear(LATENT_DIM + SPECIES_EMBED_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, species_idx):

        z = self.encode(x)

        sp_emb = self.species_embedding(species_idx)

        combined = torch.cat([z, sp_emb], dim=1)

        y_hat = self.predictor(combined)

        x_hat = self.decoder(z)

        return x_hat, y_hat, z, sp_emb


def loss_fn(x_hat, x, y_hat, y, yaw_weight=10.0):

    recon_loss = F.mse_loss(x_hat, x)

    yaw_loss = F.mse_loss(y_hat, y)

    total_loss = 0.1 * recon_loss + yaw_weight * yaw_loss

    return total_loss, recon_loss, yaw_loss


print("Revised architecture file generated successfully.")
