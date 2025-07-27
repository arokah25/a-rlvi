# inference_net.py
# Amortised corruption–probability predictor for A-RLVI
# -----------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceNet(nn.Module):
    """
    f_φ : ℝ^D  →  (0,1)
    Given backbone features z_i (and optionally a scalar context signal) returns
    π_i – the model’s belief that sample i is clean.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        use_context: bool = True,
        dropout_p: float = 0.10,
    ):
        """
        Parameters
        ----------
        input_dim   – dimensionality of backbone features (e.g. 2048 for ResNet-50)
        hidden_dim  – width of the intermediate hidden layer
        use_context – if True, a scalar context can be fused into the hidden layer
        dropout_p   – dropout rate after the hidden layer
        """
        super().__init__()
        self.use_ctx   = use_context
        self.dropout_p = dropout_p

        # 1) feature path
        self.norm = nn.LayerNorm(input_dim)
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 1)

        # 2) optional context path (linear, bias-less)
        if self.use_ctx:
            self.ctx_fc = nn.Linear(1, hidden_dim, bias=False)

    # -----------------------------------------------------------------
    def forward(
        self,
        z_i:  torch.Tensor,          # shape (B, D)
        ctx:  torch.Tensor | None = None,   # shape (B,)  or  None
    ) -> torch.Tensor:
        """
        Returns
        -------
        π_i  –  shape (B,)   probability each sample is clean
        """
        x = self.norm(z_i)                  # (B, D)

        h = F.relu(self.fc1(x))             # (B, H)

        # ---- fuse context if supplied --------------------------------
        if self.use_ctx and ctx is not None:
            # ensure ctx is 1-D (B,)  → (B,1) then project & add
            h = h + self.ctx_fc(ctx.unsqueeze(1))

        h = F.dropout(h, p=self.dropout_p, training=self.training)

        logits = self.fc2(h).squeeze(1)     # (B,)
        return torch.sigmoid(logits)        # π_i ∈ (0,1)
