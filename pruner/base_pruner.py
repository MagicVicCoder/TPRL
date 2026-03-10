from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePruner(ABC):
    """
    Abstract base class for token pruners.
    Defines the interface for pruning strategies.
    """
    def __init__(self, mllm, config):
        self.mllm = mllm
        self.config = config
        self.device = mllm.device
        self.model = None # The pruner model itself (e.g., an MLP)
        self._build_model()

    @abstractmethod
    def _build_model(self):
        """
        Build the pruner model (e.g., MLP).
        This method should populate self.model.
        """
        pass

    @abstractmethod
    def calculate_pruning_scores(self, visual_features, query_embeddings):
        """
        Calculate pruning scores for visual features based on their relevance to the query.

        Args:
            visual_features (torch.Tensor): The original visual features from the model.
                                            Shape: [batch_size, num_patches, hidden_dim]
            query_embeddings (torch.Tensor): The query embeddings representing the text prompt.
                                           Shape: [batch_size, 1, hidden_dim]

        Returns:
            torch.Tensor: Pruning scores for each patch. Higher score means less likely to be pruned.
                          Shape: [batch_size, num_patches]
        """
        pass

    def prune_tokens(self, visual_features, query_embeddings, target_ratio):
        """
        Prune visual tokens based on calculated scores.

        Args:
            visual_features (torch.Tensor): The original visual features.
                                            Shape: [batch_size, num_patches, hidden_dim]
            query_embeddings (torch.Tensor): The query embeddings.
                                           Shape: [batch_size, 1, hidden_dim]
            target_ratio (float): The ratio of tokens to keep (e.g., 0.5 for 50%).

        Returns:
            torch.Tensor: The pruned visual features.
                          Shape: [batch_size, num_patches_kept, hidden_dim]
        """
        scores = self.calculate_pruning_scores(visual_features, query_embeddings) # Shape: [B, N]

        num_patches = visual_features.shape[1]
        num_to_keep = int(num_patches * target_ratio)

        # Get indices of patches with highest scores
        _, top_indices = torch.topk(scores, num_to_keep, dim=1, largest=True, sorted=False) # Shape: [B, num_to_keep]

        batch_indices = torch.arange(visual_features.size(0)).unsqueeze(1).expand(-1, num_to_keep).to(self.device) # Shape: [B, num_to_keep]

        # Gather the pruned features
        pruned_features = visual_features[batch_indices, top_indices] # Shape: [B, num_to_keep, hidden_dim]

        return pruned_features

    def forward(self, visual_features, query_embeddings, target_ratio):
        """
        Convenience method to calculate scores and prune in one call.
        """
        return self.prune_tokens(visual_features, query_embeddings, target_ratio)

    def save_model(self, path):
        """Save the pruner model state."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load the pruner model state."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))