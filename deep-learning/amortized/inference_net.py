import torch
import torch.nn as nn

class InferenceNet(nn.Module):
    """
    A simple fully-connected inference network that takes a flattened input vector zᵢ
    and predicts a scalar corruption belief πᵢ = σ(f_ϕ(zᵢ)) ∈ (0, 1)
    """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        # z: shape (B, input_dim)
        # Output: shape (B,), each entry in (0, 1)
        return torch.sigmoid(self.net(z)).squeeze(-1)
