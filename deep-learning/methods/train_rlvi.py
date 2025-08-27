import math
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

__all__ = ["train_rlvi"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def e_step_update_pi_no_self_consistency(
    residuals: torch.Tensor,           # ℓ_i = per-sample CE from a deterministic pass
    weights: torch.Tensor,             # π from previous epoch (updated in-place)
    alpha_prev: float | None = None,   # prevalence of clean samples from prev epoch (μ_{t-1})
    eps: float = 1e-6
) -> tuple[float, float]:
    """
    Paper-aligned E-step: keep scalar offset c fixed, update π_i = σ(c - ℓ_i).
      • Choose c from α_prev via: c = log(α_prev / (1-α_prev)).
      • No inner fixed point on μ. (The self-consistent μ update collapses to 0.)
    Returns (pi_bar_after, alpha_used).
    """
     # ---- hyperparams for a stable, paper-aligned update ----
    alpha0 = 0.90      # prior clean prevalence (tune 0.85–0.95)
    rho    = 0.25      # how much we trust the prior each epoch
    clip_m = 10.0      # max CE used in E-step (cut heavy tails)
    gamma  = 1.0       # optionally <1.0 to soften: s = c - gamma*ℓ

    if alpha_prev is None:
        alpha_prev = float(weights.mean().item())

    # anchor α to a prior to prevent free-fall
    alpha_used = (1.0 - rho) * alpha_prev + rho * alpha0
    alpha_used = min(max(alpha_used, eps), 1.0 - eps)

    # scalar offset c = logit(α_used)
    c = math.log(alpha_used / (1.0 - alpha_used))

    # numerical safety on ℓ (doesn't change ordering)
    ell = residuals.clamp(min=0.0, max=clip_m)

    # vectorized π update: π_i = σ(c - γ ℓ_i)
    new_pi = torch.sigmoid(c - gamma * ell).clamp_(eps, 1.0 - eps)
    weights.copy_(new_pi)  # write back in-place

    return float(new_pi.mean().item()), float(alpha_used)



@torch.no_grad()
def _eval_residuals_fullpass(eval_loader, model, residuals):
    """
    Recompute ℓ_i for *all* training samples with θ frozen, using the
    deterministic EVAL transform loader (no aug/erasing; BN in eval mode).
    """
    was_training = model.training
    model.eval()
    for images, labels, indexes in tqdm(eval_loader, desc="RLVI E-step: eval ℓ", leave=False):
        images  = images.to(DEVICE, non_blocking=True)
        labels  = labels.to(DEVICE, non_blocking=True)
        indexes = indexes.to(DEVICE, non_blocking=True)

        logits = model(images)
        ce_vec = F.cross_entropy(logits, labels, reduction="none")
        residuals[indexes] = ce_vec
    if was_training:
        model.train()


def train_rlvi(train_loader,
               eval_loader,                     # deterministic eval transform (you already pass this)
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               residuals: torch.Tensor,         # [N] persistent CE buffer
               weights: torch.Tensor,           # [N] persistent π, updated in-place
               overfit: bool,
               threshold: float,
               writer=None,
               scheduler=None,
               epoch: int | None = None):
    """
    One epoch of RLVI:
      M-step: θ ← argmin Σ π_i ℓ_i(θ)   (use current π; normalize by Σπ for scale)
      E-step: π_i ← σ(c - ℓ_i) with c = log(α_prev / (1-α_prev)),
              where α_prev is the previous epoch's mean π (no inner μ fixed point).
    """
    N = residuals.numel()
    assert weights.shape == residuals.shape == (N,), "π and ℓ buffers must be 1-D and same length"

    # ----------------------- M-STEP -----------------------
    model.train()
    train_total = 0
    train_correct = 0
    total_loss = 0.0
    total_seen = 0

    for images, labels, indexes in tqdm(train_loader, desc=f"RLVI epoch {epoch}", leave=False):
        if torch.any(indexes < 0) or torch.any(indexes >= N):
            raise RuntimeError(f"Out-of-range indices (max={int(indexes.max())}, N={N}).")

        images  = images.to(DEVICE, non_blocking=True)
        labels  = labels.to(DEVICE, non_blocking=True)
        indexes = indexes.to(DEVICE, non_blocking=True)

        logits = model(images)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        ce_vec = F.cross_entropy(logits, labels, reduction="none")

        w = weights[indexes]
        denom = w.sum().clamp_min(1.0)            # keep grad scale stable
        loss = (ce_vec * w).sum() / denom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)

    # ---------------- recompute ℓ_i on the whole train set (frozen θ) --------
    _eval_residuals_fullpass(eval_loader, model, residuals)

    # quick sanity on epoch 1
    if epoch == 1:
        ce_min = float(residuals.min().item())
        ce_med = float(residuals.median().item())
        ce_max = float(residuals.max().item())
        mu0    = float(weights.mean().item())
        print(f"[debug:e{epoch}] CE(min/med/max) = {ce_min:.3f}/{ce_med:.3f}/{ce_max:.3f}; <π>_before = {mu0:.6f}")

    # ----------------------- E-STEP (paper-aligned) --------------------------
    alpha_prev = float(weights.mean().item())  # μ_{t-1}
    pi_bar, alpha_used = e_step_update_pi_no_self_consistency(
        residuals=residuals,
        weights=weights,
        alpha_prev=alpha_prev,
        eps=1e-6
    )

    if epoch == 1:
        print(f"[debug:e{epoch}] used α_prev={alpha_used:.6f}; <π>_after = {float(weights.mean().item()):.6f}")

    # Optional truncation if you *really* want it (off by default)
    if overfit:
        sorted_w, _ = torch.sort(weights, descending=True)
        fn_cumsum = torch.cumsum(1.0 - sorted_w, dim=0)
        beta = (1.0 - weights).sum() * 0.05
        k = (fn_cumsum <= beta).sum() - 1
        tau = sorted_w[max(k, 0)]
        weights[weights < tau] = 0.0

    train_acc = float(train_correct) / float(train_total)
    avg_loss  = total_loss / max(1, total_seen)

    if writer is not None and epoch is not None:
        writer.add_scalar("Loss/CE_weighted", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Inference/pi_mean", pi_bar, epoch)

    return train_acc, threshold, avg_loss, pi_bar
