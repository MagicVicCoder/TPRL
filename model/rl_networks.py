import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SharedAttentionModule(nn.Module):
    """
    Shared attention-based feature extractor for policy and value networks.
    Processes concatenated visual tokens and query embedding.
    """
    def __init__(self, d_model, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Query projection to match visual token dimension
        self.query_proj = nn.Linear(d_model, d_model)

        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, visual_tokens, query_embedding):
        """
        Args:
            visual_tokens: [batch_size, num_tokens, d_model]
            query_embedding: [batch_size, 1, d_model]
        Returns:
            features: [batch_size, num_tokens+1, d_model]
                     First num_tokens are visual token features, last one is query feature
        """
        # Project query
        query_proj = self.query_proj(query_embedding)  # [B, 1, d_model]

        # Concatenate visual tokens and query
        x = torch.cat([visual_tokens, query_proj], dim=1)  # [B, K+1, d_model]

        # Apply transformer
        features = self.transformer(x)  # [B, K+1, d_model]

        return features


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs retention probability for each visual token.
    """
    def __init__(self, d_model, hidden_dim=512, dropout=0.1):
        super().__init__()

        # MLP head for each token
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Output layer for retention probability
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """
        Args:
            features: [batch_size, num_tokens, d_model] (only visual token features)
        Returns:
            probs: [batch_size, num_tokens] retention probabilities
        """
        # Apply MLP with residual connection
        h = self.mlp(features)  # [B, K, hidden_dim]

        # Output retention probability
        logits = self.output(h).squeeze(-1)  # [B, K]
        probs = torch.sigmoid(logits)  # [B, K]

        return probs


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value.
    """
    def __init__(self, d_model, hidden_dim=512, dropout=0.1):
        super().__init__()

        # MLP for value estimation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        """
        Args:
            features: [batch_size, num_tokens, d_model] (only visual token features)
        Returns:
            value: [batch_size] state value
        """
        # Average pooling over tokens
        pooled = features.mean(dim=1)  # [B, d_model]

        # Estimate value
        value = self.mlp(pooled).squeeze(-1)  # [B]

        return value


class RLPruningAgent(nn.Module):
    """
    Complete RL agent with shared attention module, policy network, and value network.
    """
    def __init__(self, d_model, nhead=8, num_layers=2, hidden_dim=512, dropout=0.1):
        super().__init__()

        # Shared attention module
        self.shared_attention = SharedAttentionModule(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout
        )

        # Policy network (actor)
        self.policy = PolicyNetwork(d_model, hidden_dim, dropout)

        # Value network (critic)
        self.value = ValueNetwork(d_model, hidden_dim, dropout)

    def forward(self, visual_tokens, query_embedding, return_value=False):
        """
        Args:
            visual_tokens: [batch_size, num_tokens, d_model]
            query_embedding: [batch_size, 1, d_model]
            return_value: whether to return value estimate
        Returns:
            probs: [batch_size, num_tokens] retention probabilities
            value: [batch_size] state value (if return_value=True)
        """
        # Extract features using shared attention
        features = self.shared_attention(visual_tokens, query_embedding)  # [B, K+1, d_model]

        # Split visual token features and query feature
        visual_features = features[:, :-1, :]  # [B, K, d_model]

        # Get retention probabilities from policy
        probs = self.policy(visual_features)  # [B, K]

        if return_value:
            # Get value estimate
            value = self.value(visual_features)  # [B]
            return probs, value

        return probs

    def sample_action(self, probs, step_discount=1.0):
        """
        Sample binary actions from Bernoulli distribution.
        Args:
            probs: [batch_size, num_tokens] retention probabilities
            step_discount: lambda_disc^t factor for step-wise discounting
        Returns:
            actions: [batch_size, num_tokens] binary actions (0=prune, 1=keep)
            log_probs: [batch_size, num_tokens] log probabilities of actions
        """
        # Apply step-wise discount
        discounted_probs = probs * step_discount
        discounted_probs = torch.clamp(discounted_probs, 0.0, 1.0)

        # Sample from Bernoulli
        dist = torch.distributions.Bernoulli(discounted_probs)
        actions = dist.sample()  # [B, K]
        log_probs = dist.log_prob(actions)  # [B, K]

        return actions, log_probs

    def get_action_log_probs(self, probs, actions, step_discount=1.0):
        """
        Get log probabilities of given actions.
        """
        discounted_probs = probs * step_discount
        discounted_probs = torch.clamp(discounted_probs, 0.0, 1.0)

        dist = torch.distributions.Bernoulli(discounted_probs)
        log_probs = dist.log_prob(actions)

        return log_probs

    def get_entropy(self, probs, step_discount=1.0):
        """
        Calculate entropy of the policy distribution.
        """
        discounted_probs = probs * step_discount
        discounted_probs = torch.clamp(discounted_probs, 1e-8, 1.0 - 1e-8)

        dist = torch.distributions.Bernoulli(discounted_probs)
        entropy = dist.entropy()  # [B, K]

        return entropy.mean()
