import torch
import torch.nn as nn
import torch.nn.functional as F

class InferenceNet(nn.Module):
    """
    Inference network f_ϕ(zᵢ) that maps a feature vector zᵢ ∈ ℝᵈ 
    to a scalar πᵢ ∈ (0, 1), representing the belief that example i is clean.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Args:
            input_dim: Dimensionality of the input feature vector zᵢ (e.g., 2048 for ResNet50)
            hidden_dim: Size of the hidden layer (default: 128)
        """
        super().__init__()

        # 2-layer MLP: zᵢ → hidden → output logit → sigmoid(πᵢ)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z_i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inference network.

        Args:
            z_i: Tensor of shape (batch_size, input_dim) from model_features

        Returns:
            pi_i: Tensor of shape (batch_size,), belief scores πᵢ ∈ (0, 1)
        """
        h = F.relu(self.fc1(z_i))        # shape: (batch_size, hidden_dim)
        logits = self.fc2(h).squeeze(1)  # shape: (batch_size,)
        pi_i = torch.sigmoid(logits)     # constrain to (0, 1)

        return pi_i
