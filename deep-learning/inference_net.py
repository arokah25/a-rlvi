# inference_net.py
# Amortised corruption-probability predictor for A-RLVI
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceNet(nn.Module):
    """
    f_φ : ℝ^D  →  (0,1)
    Takes a backbone feature-vector z_i (and optionally a 1-D context
    signal) and outputs π_i – the model’s belief that sample i is clean.
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
        hidden_dim  – width of the intermediate layer
        use_context – if True, concatenate a scalar “context” input (see below)
        dropout_p   – dropout rate after the hidden layer
        """
        super().__init__()
        self.use_ctx   = use_context
        self.dropout_p = dropout_p

        # 1. Feature path
        self.norm = nn.LayerNorm(input_dim)          # keeps every dim on the same scale
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 1)

        # 2. Optional context (scalar per sample, no bias)
        if self.use_ctx:
            self.ctx_fc = nn.Linear(1, hidden_dim, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        z_i: torch.Tensor,            # (B, D)
        ctx: torch.Tensor | None = None,  # (B,) or None
    ) -> torch.Tensor:
        """
        Returns
        -------
        π_i : (B,)  – probability each sample is clean
        """
        x = self.norm(z_i)                           # (B, D), LayerNorm

        h = F.relu(self.fc1(x))                      # (B, H)
        if self.use_ctx and ctx is not None:
            h = h + self.ctx_fc(ctx.unsqueeze(1))    # add context embedding

        h = F.dropout(h, p=self.dropout_p, training=self.training)

        logits = self.fc2(h).squeeze(1)              # (B,)
        pi_i = torch.sigmoid(logits)                # (B,)
        return pi_i                 # π_i ∈ (0,1)
