import torch
import torch.nn as nn
from .base_pruner import BasePruner
from .rl_networks import RLPruningAgent

class RLPruner(BasePruner):
    """
    RL-based pruner using policy network for adaptive token pruning.
    """
    def __init__(self, mllm, config):
        self.use_autoencoder = getattr(config, 'USE_AUTOENCODER', False)
        super().__init__(mllm, config)

        # RL-specific parameters
        self.threshold = getattr(config, 'RL_THRESHOLD', 0.5)
        self.step_discount = getattr(config, 'RL_STEP_DISCOUNT', 0.5)

    def _build_model(self):
        """
        Build the RL agent (policy + value networks).
        """
        # Get feature dimension from MLLM
        feature_dim = self.mllm.feature_dim

        if self.use_autoencoder:
            # TODO: Use compressed latent dimension
            latent_dim = getattr(self.config, 'RL_LATENT_DIM', 256)
            input_dim = latent_dim
        else:
            # Use original visual token dimension
            input_dim = feature_dim

        # Create RL agent
        self.model = RLPruningAgent(
            d_model=input_dim,
            nhead=getattr(self.config, 'RL_NHEAD', 8),
            num_layers=getattr(self.config, 'RL_NUM_LAYERS', 2),
            hidden_dim=getattr(self.config, 'RL_HIDDEN_DIM', 512),
            dropout=getattr(self.config, 'RL_DROPOUT', 0.1)
        ).to(self.device)

        if self.use_autoencoder:
            # TODO: Load autoencoder
            self.autoencoder = None
        else:
            self.autoencoder = None

    def calculate_pruning_scores(self, visual_features, query_embeddings):
        """
        Calculate pruning scores using the policy network.

        Args:
            visual_features: [batch_size, num_patches, hidden_dim]
            query_embeddings: [batch_size, 1, hidden_dim]
        Returns:
            scores: [batch_size, num_patches] retention probabilities
        """
        with torch.no_grad():
            # Compress features if using autoencoder
            if self.use_autoencoder and self.autoencoder is not None:
                # TODO: Apply autoencoder
                compressed_features = self.autoencoder.encode(visual_features)
            else:
                compressed_features = visual_features

            # Get retention probabilities from policy
            probs = self.model(compressed_features, query_embeddings, return_value=False)

        return probs

    def prune_tokens_deterministic(self, visual_features, query_embeddings, threshold=None):
        """
        Deterministic one-shot pruning for inference.

        Args:
            visual_features: [batch_size, num_patches, hidden_dim]
            query_embeddings: [batch_size, 1, hidden_dim]
            threshold: retention probability threshold (default: self.threshold)
        Returns:
            pruned_features: [batch_size, num_kept, hidden_dim]
        """
        if threshold is None:
            threshold = self.threshold

        # Get retention probabilities
        probs = self.calculate_pruning_scores(visual_features, query_embeddings)  # [B, N]

        # Deterministic thresholding
        mask = (probs > threshold).float()  # [B, N]

        # Apply mask to original visual features
        batch_size, num_patches, hidden_dim = visual_features.shape

        # Ensure at least one token is kept per sample
        for b in range(batch_size):
            if mask[b].sum() == 0:
                # Keep the token with highest probability
                max_idx = probs[b].argmax()
                mask[b, max_idx] = 1.0

        # Gather kept tokens
        kept_indices = []
        pruned_features_list = []

        for b in range(batch_size):
            indices = torch.where(mask[b] > 0)[0]
            kept_indices.append(indices)
            pruned_features_list.append(visual_features[b, indices, :])

        # Pad to same length for batching
        max_kept = max(len(indices) for indices in kept_indices)
        pruned_features = torch.zeros(batch_size, max_kept, hidden_dim, device=self.device)

        for b in range(batch_size):
            num_kept = len(kept_indices[b])
            pruned_features[b, :num_kept, :] = pruned_features_list[b]

        return pruned_features

    def prune_tokens(self, visual_features, query_embeddings, target_ratio):
        """
        Prune tokens using top-k selection based on retention probabilities.
        This is used during evaluation.

        Args:
            visual_features: [batch_size, num_patches, hidden_dim]
            query_embeddings: [batch_size, 1, hidden_dim]
            target_ratio: ratio of tokens to keep
        Returns:
            pruned_features: [batch_size, num_kept, hidden_dim]
        """
        scores = self.calculate_pruning_scores(visual_features, query_embeddings)

        num_patches = visual_features.shape[1]
        num_to_keep = int(num_patches * target_ratio)
        num_to_keep = max(1, num_to_keep)  # Keep at least 1 token

        # Get indices of top-k tokens
        _, top_indices = torch.topk(scores, num_to_keep, dim=1, largest=True, sorted=False)

        batch_indices = torch.arange(visual_features.size(0)).unsqueeze(1).expand(-1, num_to_keep).to(self.device)

        # Gather pruned features
        pruned_features = visual_features[batch_indices, top_indices]

        return pruned_features

    def forward(self, visual_features, query_embeddings, target_ratio=None, use_threshold=False):
        """
        Forward pass for pruning.

        Args:
            visual_features: [batch_size, num_patches, hidden_dim]
            query_embeddings: [batch_size, 1, hidden_dim]
            target_ratio: ratio of tokens to keep (if not using threshold)
            use_threshold: whether to use threshold-based pruning
        Returns:
            pruned_features: [batch_size, num_kept, hidden_dim]
        """
        if use_threshold:
            return self.prune_tokens_deterministic(visual_features, query_embeddings)
        else:
            if target_ratio is None:
                target_ratio = self.config.PRUNING_TARGET_RATIO
            return self.prune_tokens(visual_features, query_embeddings, target_ratio)
