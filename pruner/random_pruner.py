import torch
import torch.nn as nn
from .base_pruner import BasePruner
import random

class RandomPruner(BasePruner):
    """
    Pruner that randomly removes visual tokens.
    This serves as a baseline comparison against learned pruning methods.
    """
    def _build_model(self):
        # Random pruner doesn't require a learnable model.
        # We can set self.model to None or a placeholder if the base class requires it.
        # For this implementation, we'll just pass, as the pruning logic doesn't rely on a neural net.
        print("Random Pruner initialized (no model to build).")

    def calculate_pruning_scores(self, visual_features, query_embeddings):
        """
        For random pruning, we assign random scores to each patch.
        This effectively means the top-k selection becomes random.

        Args:
            visual_features (torch.Tensor): Shape [B, N, hidden_dim]
            query_embeddings (torch.Tensor): Shape [B, 1, hidden_dim] (Not used in random pruning)

        Returns:
            torch.Tensor: Random pruning scores for each patch.
                          Shape: [B, N]
        """
        batch_size, num_patches, _ = visual_features.shape
        # Generate random scores (e.g., from a uniform distribution)
        # Using torch.rand ensures scores are different for each call and patch
        random_scores = torch.rand(batch_size, num_patches, device=self.device)
        return random_scores

    # The 'prune_tokens' and 'forward' methods from BasePruner can be inherited
    # as they use 'calculate_pruning_scores' internally.
    # Alternatively, you can override 'prune_tokens' directly for potentially more control:
    # def prune_tokens(self, visual_features, query_embeddings, target_ratio):
    #     """
    #     Prune visual tokens randomly.
    #     """
    #     batch_size, num_patches, hidden_dim = visual_features.shape
    #     num_to_keep = int(num_patches * target_ratio)
    #
    #     if num_to_keep >= num_patches:
    #         return visual_features # Return all if target ratio is 1.0 or more
    #
    #     # Generate random indices for each batch item
    #     pruned_features_list = []
    #     for i in range(batch_size):
    #         # Create a list of indices and shuffle them
    #         indices = list(range(num_patches))
    #         random.shuffle(indices) # Or use torch.randperm for tensor operations
    #         selected_indices = sorted(indices[:num_to_keep]) # Keep sorted for consistent order
    #         selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
    #         pruned_features_list.append(visual_features[i, selected_indices_tensor, :].unsqueeze(0)) # Add batch dim back
    #
    #     return torch.cat(pruned_features_list, dim=0)
    # Using the inherited method with random scores is simpler and equivalent.
