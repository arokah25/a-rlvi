import torch
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm.auto import tqdm

__all__ = ["train_rlvi"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def update_sample_weights(residuals: torch.Tensor,
                          weights: torch.Tensor,
                          tol: float = 1e-3,
                          maxiter: int = 40):
    """
    Colleague-style E-step:
      1) shift CE to be non-negative: ℓ <- ℓ - min(ℓ)
      2) exp(-ℓ)
      3) iterate avg_weight (mu) via logistic mapping with ratio = mu/(1-mu)
      4) stop when ||π_new-π|| < tol
      5) rescale to [0,1] by dividing by max (note: this inflates the mean)
    """
    # 1) shift CE so min is 0 (keeps ordering; reduces blow-up)
    residuals.sub_(residuals.min())

    # 2) exponentiate the "goodness"
    exp_res = torch.exp(-residuals)

    # 3) iterate avg_weight
    avg_weight = 0.95  # their starting prevalence
    for _ in range(maxiter):
        ratio = avg_weight / (1.0 - avg_weight)
        new_weights = (ratio * exp_res) / (1.0 + ratio * exp_res)

        error = torch.norm(new_weights - weights)
        weights.copy_(new_weights)
        avg_weight = float(weights.mean().item())
        if error < tol:
            break

    # 5) rescale to [0,1] by max (faithful to their code)
    wmax = float(weights.max().item())
    if wmax > 0:
        weights.div_(wmax)


def false_negative_criterion(weights: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Find threshold from fixed probability (alpha) of type II error
    (identical to your colleague’s implementation).
    """
    total_positive = torch.sum(1.0 - weights)
    beta = total_positive * alpha
    sorted_weights, _ = torch.sort(weights, dim=0, descending=True)
    false_negative = torch.cumsum(1.0 - sorted_weights, dim=0)
    last_index = torch.sum(false_negative <= beta) - 1
    threshold = sorted_weights[max(int(last_index.item()), 0)]
    return threshold


def train_rlvi(train_loader,
               eval_loader,                     # NOTE: accepted but unused (keeps main.py intact)
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
    One epoch of the colleague-style RLVI (as pasted):
      • M-step: update θ with per-sample CE weighted by current π[indexes]
                while ALSO writing current batch CE into `residuals[indexes]`.
      • E-step: call `update_sample_weights(residuals, weights)` (in-place).
      • Optional truncation via type II criterion when `overfit` is True.

    Returns:
      train_acc, threshold, avg_loss, pi_bar
    """
    model.train()

    train_total = 0
    train_correct = 0
    total_loss = 0.0
    total_seen = 0

    # --- M-step (SGD with current π) + write residuals like colleague’s code ---
    for images, labels, indexes in tqdm(train_loader, desc=f"RLVI epoch {epoch}", leave=False):
        images  = Variable(images).to(DEVICE, non_blocking=True)
        labels  = Variable(labels).to(DEVICE, non_blocking=True)
        indexes = indexes.to(DEVICE, non_blocking=True)

        logits = model(images)

        # Top-1 accuracy in their style (per-batch, averaged)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            # They used `accuracy` that returns percentage as a scalar tensor per batch.
            # We mimic that behavior by averaging per-batch (so epoch accuracy is mean of batch accuracies).
            correct = (preds == labels).float().mean()  # in [0,1]
            train_total += 1
            train_correct += float(correct.item())

        # Per-sample CE
        ce_vec = F.cross_entropy(logits, labels, reduction="none")

        # Save the raw losses for the E-step (colleague does it inside M-step)
        residuals[indexes] = ce_vec.detach()

        # Weight the loss with current π
        w = weights[indexes]
        loss = (ce_vec * w).mean()  # colleague uses mean, not normalization by sum(w)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item()) * images.size(0)
        total_seen += int(images.size(0))

    # --- E-step (colleague style) ---
    update_sample_weights(residuals, weights)

    # --- Optional truncation (unchanged) ---
    if overfit:
        threshold = max(float(threshold), float(false_negative_criterion(weights)))
        weights[weights < threshold] = 0.0

    train_acc = float(train_correct) / max(1, train_total)  # mean of batch accuracies
    avg_loss  = total_loss / max(1, total_seen)
    pi_bar    = float(weights.mean().item())

    # logging hooks, kept compatible with your main.py
    if writer is not None and epoch is not None:
        writer.add_scalar("Loss/CE_weighted_mean", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy_epochmean", train_acc, epoch)
        writer.add_scalar("Inference/pi_mean", pi_bar, epoch)

    return train_acc, threshold, avg_loss, pi_bar
