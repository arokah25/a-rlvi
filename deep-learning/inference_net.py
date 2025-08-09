# inference_net.py
# Amortised corruption–probability predictor for A-RLVI (no context branch)
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_squash_to_open_unit_interval(logits: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """
    Map logits -> (eps, 1 - eps). This avoids π hitting 0/1 where gradients die.
    π = eps + (1 - 2*eps) * sigmoid(logits)
    """
    sig = torch.sigmoid(logits)
    return eps + (1.0 - 2.0 * eps) * sig


class InferenceNet(nn.Module):
    """
    f_φ : R^D -> (0,1)
    Given backbone features z, returns π — the model’s belief that the sample is clean.

    Usage note (enforced in the training loop, not here):
      For the φ-loss (KL(π‖q(τ)) + KL(π‖π̄)), call with DETACHED features:
          pi = inference_net(z.detach())
      so φ’s loss does not update the backbone via z.
    """

    def __init__(
        self,
        input_dim:  int,    # e.g., 2048 for ResNet-50 GAP features
        hidden_dim: int = 64,
        dropout_p:  float = 0.10,
        init_pi:    float = 0.5,   # prior belief; bias init uses logit(init_pi)
        use_silu:   bool  = True,  # SiLU tends to be a bit smoother than ReLU
        eps_margin: float = 0.05,  # output kept in (eps, 1-eps); fixed, not a knob
    ):
        super().__init__()
        assert 0.0 < init_pi   < 1.0,  "init_pi must be in (0,1)"
        assert 0.0 < eps_margin < 0.5, "eps_margin must be in (0, 0.5)"

        self.dropout_p  = float(dropout_p)
        self.eps_margin = float(eps_margin)
        self.act        = nn.SiLU() if use_silu else nn.ReLU(inplace=True)

        # Architecture: LayerNorm -> Linear -> act -> Dropout -> Linear -> π
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=True)
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 1)

        # ---- Initialization -------------------------------------------------
        # Hidden: Kaiming for ReLU/SiLU
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        # Output: start near the prior via bias; keep weights small so bias dominates initially
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.fill_(torch.logit(torch.tensor(init_pi)))

        # (Perhaps) add weight norm to fc2 for extra stability:
        # self.fc2 = nn.utils.weight_norm(self.fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, D) backbone features (detach at callsite for φ-loss)
        Returns:
          pi : (B,) probability each sample is clean, in (eps_margin, 1 - eps_margin)
        """
        x = self.norm(z)                # (B, D)
        h = self.act(self.fc1(x))       # (B, H)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        logits = self.fc2(h).squeeze(1) # (B,)
        pi = _soft_squash_to_open_unit_interval(logits, eps=self.eps_margin)
        return pi
