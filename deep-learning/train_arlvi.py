"""
train_arlvi.py
==============

Mini-batch training loop for A-RLVI with practical-stability fixes:

  1. Soft squashing of πᵢ ∈ (0.05,0.95) keeps gradients alive.
  2. Weighted cross-entropy is **normalised by Σ πᵢ** to keep loss scale stable.
  3. KL prior π̄ is detached from the graph (no gradient into the prior).
  4. π̄ is updated by an EMA **once per epoch** – gives a true anchor.
  5. Entropy regularisation is *subtracted* (encourages uncertainty).
  6. Gradient clipping on the inference net.

The function returns epoch-level metrics and the updated π̄ₑₘₐ value so
`main.py` can feed it into the next epoch.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Helper: KL( Bern(pi_i) || Bern(pi_bar) )  element-wise
# ---------------------------------------------------------------------
def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence between two Bernoulli distributions.

    Args
    ----
    pi_i   : (B,)  posterior probabilities
    pi_bar : (B,)  prior probabilities, same shape

    Returns
    -------
    kl     : (B,)  KL divergences for each sample
    """
    eps = 1e-6
    pi_i   = pi_i.clamp(eps, 1.0 - eps)
    pi_bar = pi_bar.clamp(eps, 1.0 - eps)
    kl = pi_i * torch.log(pi_i / pi_bar) + (1.0 - pi_i) * torch.log((1.0 - pi_i) / (1.0 - pi_bar))
    return kl


# ---------------------------------------------------------------------
# Main epoch routine
# ---------------------------------------------------------------------
def train_arlvi(
    model_features:      torch.nn.Module,   # frozen / finetuned CNN backbone
    model_classifier:    torch.nn.Module,   # classifier head
    inference_net:       torch.nn.Module,   # amortised corruption network f_φ
    dataloader:          torch.utils.data.DataLoader,
    optimizer:           torch.optim.Optimizer,      # for θ (features+classifier)
    inference_optimizer: torch.optim.Optimizer,      # for φ
    device:              torch.device,
    epoch:               int,
    *,
    lambda_kl:     float = 1.0,   # weight for KL term
    pi_bar:        float = 0.9,   # warm-up prior value
    warmup_epochs: int   = 2,
    alpha:         float = 0.97,  # EMA momentum for π̄ after warm-up
    pi_bar_ema:    float = 0.9,   # running prior coming in from previous epoch
    beta:          float = 0.1,   # weight on entropy regularisation
    writer=None,                  # optional TensorBoard writer
    grad_clip:     float = 5.0,   # clip on inference-net gradients (None = off)
):
    """
    One training epoch of A-RLVI.

    Returns
    -------
    avg_ce_loss : mean normalised CE over epoch
    avg_kl_loss : mean KL over epoch
    train_acc   : accuracy on training data
    mean_pi_i   : mean posterior probability over epoch
    pi_bar_ema  : updated EMA prior (to pass back to caller)
    """

    # -----------------------------------------------------------------
    # Put models in train mode
    # -----------------------------------------------------------------
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # -----------------------------------------------------------------
    # Running statistics
    # -----------------------------------------------------------------
    total_loss = total_ce = total_kl = 0.0
    total_correct = total_seen = 0
    all_pi_values = []        # used for histogram + epoch mean

    eps = 1e-8                # numerical safety for logs / div

    # -----------------------------------------------------------------
    # CONSTANT prior tensor for this epoch’s batches
    #  - warm-up   : fixed optimistic prior (pi_bar)
    #  - finetune  : detached tensor filled with π̄ₑₘₐ from last epoch
    # -----------------------------------------------------------------
    if epoch < warmup_epochs:
        prior_value = torch.tensor(pi_bar, device=device)
    else:
        prior_value = torch.tensor(pi_bar_ema, device=device)

    # -----------------------------------------------------------------
    # Mini-batch loop
    # -----------------------------------------------------------------
    for batch_idx, (images, labels, *_ ) in enumerate(dataloader):
        # ------------- data to device --------------------------------
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # ------------- forward pass (θ) ------------------------------
        z_i = model_features(images)           # (B,C,1,1)
        z_i = z_i.view(B, -1)                  # flatten → (B,C)
        logits = model_classifier(z_i)         # (B,num_classes)

        # ------------- posterior πᵢ  (soft squash) -------------------
        pi_raw = inference_net(z_i)            # sigmoid inside net → (0,1)
        pi_i   = 0.9 * pi_raw + 0.05           # (0.05,0.95) keeps grad alive

        all_pi_values.append(pi_i.detach().cpu())

        # ------------- per-sample CE --------------------------------
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)

        # ------------- KL divergence --------------------------------
        pi_bar_tensor = torch.full_like(pi_i, prior_value).detach()  # no grad
        kl_loss = compute_kl_divergence(pi_i, pi_bar_tensor)         # (B,)

        # ------------- Entropy regularisation -----------------------
        entropy = -(pi_i*torch.log(pi_i+eps) + (1-pi_i)*torch.log(1-pi_i+eps))
        entropy_reg = beta * entropy.mean()          # subtract later

        # ------------- Loss composition -----------------------------
        ce_weighted = (pi_i * ce_loss).sum() / (pi_i.sum() + eps)
        mean_kl     = kl_loss.mean()

        total_loss_batch = ce_weighted + lambda_kl * mean_kl - entropy_reg

        # ------------- Back-prop ------------------------------------
        optimizer.zero_grad()
        inference_optimizer.zero_grad()
        total_loss_batch.backward()

        # gradient clip for inference net (helps when π collapses)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(inference_net.parameters(), grad_clip)

        optimizer.step()
        inference_optimizer.step()

        # ------------- Stats ----------------------------------------
        total_loss += total_loss_batch.item() * B
        total_ce   += ce_weighted.item()      * B
        total_kl   += mean_kl.item()          * B
        total_seen += B

        preds = logits.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()

        # ---- debug every 500 batches -------------------------
        if batch_idx % 500 == 0:
            # Measure grad norm WITHOUT a second clip
            grad_inf = torch.nn.utils.clip_grad_norm_(
                inference_net.parameters(), float('inf')
            ).item()

            print(f"[Epoch {epoch:02d} Batch {batch_idx:04d}] "
                  f"πᵢ μ={pi_i.mean():.3f} min={pi_i.min():.2f} max={pi_i.max():.2f} "
                  f"CE={ce_weighted.item():.3f}  KL={mean_kl.item():.3f}  "
                  f"|∇φ|={grad_inf:.2f}")

    # ---------------- end mini-batch loop ---------------------------

    # -----------------------------------------------------------------
    # Once-per-epoch EMA update of the prior
    # -----------------------------------------------------------------
    mean_pi_i = torch.cat(all_pi_values).mean().item()
    if epoch >= warmup_epochs:
        pi_bar_ema = alpha * pi_bar_ema + (1.0 - alpha) * mean_pi_i

    # -----------------------------------------------------------------
    # Aggregate epoch metrics
    # -----------------------------------------------------------------
    avg_loss    = total_loss / total_seen
    avg_ce_loss = total_ce   / total_seen
    avg_kl_loss = total_kl   / total_seen
    train_acc   = total_correct / total_seen

    # -------- TensorBoard logging ------------------------
    if writer is not None:
        writer.add_scalar("Loss/CE_weighted", avg_ce_loss, epoch)
        writer.add_scalar("Loss/KL",          avg_kl_loss, epoch)
        writer.add_scalar("Inference/pi_bar_ema", pi_bar_ema, epoch)
        if epoch % 10 == 0:
            pi_hist = torch.cat(all_pi_values)
            writer.add_histogram("Inference/PiDistribution", pi_hist, epoch)
            writer.flush()

    # -----------------------------------------------------------------
    return avg_ce_loss, avg_kl_loss, train_acc, mean_pi_i, pi_bar_ema
