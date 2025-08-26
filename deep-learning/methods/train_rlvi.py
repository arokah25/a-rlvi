import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

__all__ = ["train_rlvi"]

# Use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def fixed_point_update_pi(residuals: torch.Tensor,
                          weights: torch.Tensor,
                          tol: float = 1e-6,
                          maxiter: int = 100) -> float:
    """
    E-step (RLVI) as a *fixed-point iteration* that *converges* each epoch.

    Paper alignment:
      • We use the closed-form fixed-point update (Eq. 18) with ε eliminated
        via ε* = 1 - <π>. Algebraically:
            π_i = σ( logit(<π>) - ℓ_i )
        where ℓ_i is the *per-sample* negative log-likelihood (CE here),
        σ is sigmoid, and <π> is the *current* mean of π in the inner loop.
      • We iterate until π stabilizes (self-consistency), which is what the
        paper's Algorithm 2 implies by “fixed-point iterations (18)”.

    Args
    ----
    residuals : Tensor [N]
        Per-sample CE losses ℓ_i collected over the *current* model θ
        on the training set during this epoch (reduction='none').
    weights   : Tensor [N], in-place
        π vector from the previous epoch; will be replaced by the converged π.
    tol       : float
        L1 convergence tolerance on π between inner iterations.
    maxiter   : int
        Safety cap on inner iterations.

    Returns
    -------
    pi_bar : float
        The converged mean π̄ = <π> (for logging).
    """
    # Start inner iteration from the previous epoch's π (Algorithm 2 init)
    pi = weights.clamp(1e-8, 1 - 1e-8)

    # IMPORTANT: do *not* shift residuals; the update depends on true ℓ_i
    # (shifting would implicitly change the prior ratio).

    for _ in range(maxiter):
        # Current <π> (must stay in (0,1) to keep logit finite)
        mu = pi.mean().clamp(1e-6, 1 - 1e-6)

        # logit(<π>) - ℓ_i  (Eq. 18 in a numerically stable form)
        s = torch.log(mu / (1.0 - mu)) - residuals

        # π_new = σ(s)
        new_pi = torch.sigmoid(s).clamp(1e-8, 1 - 1e-8)

        # Convergence check (L1 mean)
        if torch.mean(torch.abs(new_pi - pi)) < tol:
            pi.copy_(new_pi)
            break

        pi.copy_(new_pi)

    # Write back to the shared weights vector used next epoch
    weights.copy_(pi)
    return float(pi.mean().item())


@torch.no_grad()
def false_negative_criterion(weights: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Optional truncation used in the paper to reduce false negatives.
    Not needed for Food-101 per their appendix (but kept here as a hook).

    Returns a threshold τ such that setting π_i=0 when π_i<τ bounds type-II error.
    """
    total_positive = torch.sum(1 - weights)
    beta = total_positive * alpha
    sorted_weights, _ = torch.sort(weights, dim=0, descending=True)
    false_negative = torch.cumsum(1 - sorted_weights, dim=0)
    last_index = torch.sum(false_negative <= beta) - 1
    threshold = sorted_weights[last_index]
    return threshold


def train_rlvi(train_loader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               residuals: torch.Tensor,    # [N] persistent buffer of CE per sample
               weights: torch.Tensor,      # [N] persistent vector of π (sample weights)
               overfit: bool,
               threshold: float,
               writer=None,
               epoch: int = None):
    """
    One training epoch of RLVI with E/M alternation:

      M-step during the epoch:
        – Use *previous* epoch's π (weights) to minimize ∑_i π_i * ℓ_i(θ)
          via standard SGD/Adam on mini-batches.

      E-step *after* the epoch:
        – With residuals (per-sample CE) computed for current θ, run the
          *converged* fixed-point update for π (Eq. 18) until self-consistent.

    Returns
    -------
    train_acc : float
        Top-1 training accuracy (count-based).
    threshold : float
        Updated truncation threshold (only used if `overfit=True`).
    avg_loss  : float
        Average *weighted* CE over the epoch.
    pi_bar    : float
        Mean π after the converged E-step (for logging).
    """
    model.train()

    train_total = 0
    train_correct = 0
    total_loss = 0.0
    total_seen = 0

    # ---- M-step (use *current* π to train θ for one pass) ------------------
    progress = tqdm(train_loader, desc=f"RLVI epoch {epoch}", leave=False)
    for images, labels, indexes in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)

        # Count-based accuracy for readability/robustness
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # Per-sample CE (NEG log-likelihood); keep reduction='none'
        ce_vec = F.cross_entropy(logits, labels, reduction="none")

        # Store *raw* CE for E-step (whole-epoch buffers)
        residuals[indexes] = ce_vec.detach()

        # Weighted loss with current π
        batch_weights = weights[indexes]
        loss = (ce_vec * batch_weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)

        # Live progress within the epoch
        running_acc = 100.0 * train_correct / max(1, train_total)
        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            acc=f"{running_acc:.2f}%",
            pi_mean=f"{weights.mean().item():.3f}"
        )

    # ---- E-step (converged fixed-point update of π given current residuals) -
    pi_bar = fixed_point_update_pi(residuals=residuals, weights=weights, tol=1e-6, maxiter=100)

    # Optional truncation (not recommended for Food-101 per paper’s appendix)
    if overfit:
        threshold = max(threshold, float(false_negative_criterion(weights)))
        weights[weights < threshold] = 0.0

    # Epoch summaries
    train_acc = float(train_correct) / float(train_total)
    avg_loss = total_loss / total_seen

    # TensorBoard (optional)
    if writer is not None and epoch is not None:
        writer.add_scalar("Loss/CE_weighted", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Inference/pi_mean", pi_bar, epoch)

    return train_acc, threshold, avg_loss, pi_bar
