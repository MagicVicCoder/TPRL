import torch
import torch.nn as nn
from .base_pruner import BasePruner

class MLPPruner(BasePruner):
    """
    Pruner using a simple Multi-Layer Perceptron (MLP) to predict pruning scores.
    The MLP takes visual features and query embeddings as input to score each patch.
    """
    def _build_model(self):
        hidden_dim = self.mllm.feature_dim
        mlp_hidden_dim = self.config.PRUNING_MLP_HIDDEN_DIM
        dropout_rate = self.config.PRUNING_MLP_DROPOUT

        # Simple MLP: Concatenated features -> Hidden -> ReLU -> Hidden -> ReLU -> Score
        # Input per patch: [visual_feature, query_feature] -> [2 * hidden_dim]
        # Output per patch: [1] (score)
        self.model = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, 1)
        ).to(self.device)
        print("MLP Pruner model built successfully.")

    def calculate_pruning_scores(self, visual_features, query_embeddings):
        """
        Calculate pruning scores using the MLP.

        Args:
            visual_features (torch.Tensor): Shape [B, N, hidden_dim]
            query_embeddings (torch.Tensor): Shape [B, 1, hidden_dim]

        Returns:
            torch.Tensor: Shape [B, N] - Pruning scores for each patch.
        """
        batch_size, num_patches, hidden_dim = visual_features.shape

        # Expand query to match visual features shape [B, N, hidden_dim]
        expanded_query = query_embeddings.expand(-1, num_patches, -1)

        # Concatenate visual and query features [B, N, 2 * hidden_dim]
        combined_features = torch.cat([visual_features, expanded_query], dim=-1)

        # Reshape for MLP: [B * N, 2 * hidden_dim]
        combined_features_flat = combined_features.view(-1, 2 * hidden_dim)

        # Get raw scores [B * N, 1]
        raw_scores_flat = self.model(combined_features_flat)

        # Reshape back to [B, N]
        scores = raw_scores_flat.view(batch_size, num_patches)

        # Optionally, apply sigmoid to bound scores between 0 and 1, or use raw scores
        # Here we use raw scores directly, higher score means more important
        return scores