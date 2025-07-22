"""
train_arlvi.py
==============

Mini-batch training loop for A-RLVI with practical-stability fixes:

  1. Soft squashing of πᵢ ∈ (0.05,0.95) keeps gradients alive.
  2. Rubber-band KL weight starts at 4.0, decays to 1.0 over 5 epochs.
  3. KL prior π̄ is detached from the graph (no gradient into the prior).
  4. π̄ is updated by an EMA **once per epoch** and is detached from the graph.
  5. Entropy regularisation is *subtracted* (encourages uncertainty).
     – it is linearly annealed from β=0.4 to 0.0 over epochs 12-22.
  6. Gradient clipping on the inference net.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Helper: KL( Bern(pi_i) || Bern(pi_bar) )  element-wise
# ---------------------------------------------------------------------
def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL between two Bernoulli distributions (batched)."""
    eps = 1e-6
    pi_i   = pi_i.clamp(eps, 1. - eps)
    pi_bar = pi_bar.clamp(eps, 1. - eps)
    return pi_i*torch.log(pi_i/pi_bar) + (1.-pi_i)*torch.log((1.-pi_i)/(1.-pi_bar))


# ---------------------------------------------------------------------
# Main epoch routine
# ---------------------------------------------------------------------
def train_arlvi(
    model_features:      torch.nn.Module,
    model_classifier:    torch.nn.Module,
    inference_net:       torch.nn.Module,
    dataloader:          torch.utils.data.DataLoader,
    optimizer:           dict,                     # {"backbone": …, "classifier": …}
    inference_optimizer: torch.optim.Optimizer,    # for φ
    device:              torch.device,
    epoch:               int,
    *,
    lambda_kl:     float = 1.0,
    pi_bar:        float = 0.9,
    warmup_epochs: int   = 2,
    alpha:         float = 0.97,
    pi_bar_ema:    float = 0.9,
    beta:          float = 0.4,
    tau:           float = 0.6,
    scheduler=None,                    # {"backbone": …, "classifier": …} or None
    writer=None,
    grad_clip:     float = 5.0,
):
    # -----------------------------------------------------------------
    # Modes & bookkeeping
    # -----------------------------------------------------------------
    model_features.train()
    model_classifier.train()
    inference_net.train()

    total_loss = total_ce = total_kl = 0.0
    total_correct = total_seen = 0
    all_pi_values = []
    eps = 1e-8

    prior_value = torch.tensor(pi_bar if epoch < warmup_epochs else pi_bar_ema,
                               device=device)

    # -----------------------------------------------------------------
    # Mini-batch loop
    # -----------------------------------------------------------------
    for batch_idx, (images, labels, *_ ) in enumerate(dataloader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        B = images.size(0)

        # Forward pass (θ)
        z_i     = model_features(images).view(B, -1)
        logits  = model_classifier(z_i)

        # Posterior πᵢ with gradual squash ramp
        if epoch < warmup_epochs:
            with torch.no_grad():
                pi_raw = inference_net(z_i)
        else:
            pi_raw = inference_net(z_i)

        ramp_T  = 6
        t       = min(epoch, ramp_T) / ramp_T
        scale   = 0.4 + 0.55 * t          # 0.40 → 0.95
        offset  = 0.5 - scale / 2         # 0.30 → 0.025
        pi_i    = scale * pi_raw + offset
        all_pi_values.append(pi_i.detach().cpu())

        # Loss terms
        ce_loss   = F.cross_entropy(logits, labels, reduction='none')
        kl_loss   = compute_kl_divergence(pi_i, prior_value.expand_as(pi_i))
        pi_temp   = pi_i ** tau
        ce_weight = (pi_temp * ce_loss).sum() / B
        mean_kl   = kl_loss.mean()

        # Entropy reg with linear anneal
        decay_start, decay_len = 12, 22
        beta_now = beta * max(0., 1. - max(epoch - decay_start, 0) / decay_len)
        entropy_reg = beta_now * (-(pi_i*torch.log(pi_i+eps) +
                                    (1-pi_i)*torch.log(1-pi_i+eps))).mean()

        # Rubber-band KL weight
        epochs_after = max(epoch - warmup_epochs, 0)
        kl_weight    = max(lambda_kl, 6 - 0.6 * epochs_after)

        total_batch_loss = ce_weight + kl_weight*mean_kl - entropy_reg

        # -----------------------------------------------------------------
        # zero-grad, backward, clip, step
        # -----------------------------------------------------------------
        optim_backbone   = optimizer["backbone"]
        optim_classifier = optimizer["classifier"]

        optim_backbone.zero_grad(set_to_none=True)
        if epoch >= warmup_epochs:
            optim_classifier.zero_grad(set_to_none=True)
            inference_optimizer.zero_grad(set_to_none=True)

        total_batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model_features.parameters(),   5.0)
        if epoch >= warmup_epochs:
            torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(inference_net.parameters(),   grad_clip)

        optim_backbone.step()
        if epoch >= warmup_epochs:
            optim_classifier.step()
            inference_optimizer.step()

        # -----------------------------------------------------------------
        # schedulers
        # -----------------------------------------------------------------
        if scheduler is not None:
            scheduler["backbone"].step()
            if epoch >= warmup_epochs:             # ← stepped only once updates begin
                scheduler["classifier"].step()

            if writer is not None:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("LR/backbone",   scheduler["backbone"].get_last_lr()[0],   step)
                if epoch >= warmup_epochs:
                    writer.add_scalar("LR/classifier", scheduler["classifier"].get_last_lr()[0], step)

        # -----------------------------------------------------------------
        # Stats
        # -----------------------------------------------------------------
        total_loss   += total_batch_loss.item() * B
        total_ce     += ce_weight.item()       * B
        total_kl     += mean_kl.item()         * B
        total_seen   += B
        total_correct += logits.argmax(1).eq(labels).sum().item()

        if batch_idx % 500 == 0:
            grad_inf = torch.nn.utils.clip_grad_norm_(inference_net.parameters(), float('inf')).item()
            clf_norm = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), float('inf')).item()
            print(f"[Epoch {epoch:02d} Batch {batch_idx:04d}] "
                  f"πᵢ μ={pi_i.mean():.3f} min={pi_i.min():.2f} max={pi_i.max():.2f} "
                  f"CE={ce_weight.item():.3f} KL={mean_kl.item():.3f} "
                  f"|∇φ|={grad_inf:.2f} |∇θ|={clf_norm:.2f}")

    # -----------------------------------------------------------------
    # Once-per-epoch EMA update of the prior
    # -----------------------------------------------------------------
    mean_pi_i = torch.cat(all_pi_values).mean().item()
    if epoch >= warmup_epochs:
        pi_bar_ema = alpha * pi_bar_ema + (1. - alpha) * mean_pi_i

    # -----------------------------------------------------------------
    # Epoch aggregates
    # -----------------------------------------------------------------
    avg_ce_loss = total_ce / total_seen
    avg_kl_loss = total_kl / total_seen
    train_acc   = total_correct / total_seen

    if writer is not None:
        writer.add_scalar("Loss/CE_weighted",       avg_ce_loss, epoch)
        writer.add_scalar("Loss/KL",                avg_kl_loss, epoch)
        writer.add_scalar("Inference/pi_bar_ema",   pi_bar_ema,  epoch)
        if epoch % 3 == 0:
            writer.add_histogram("Inference/PiDistribution",
                                 torch.cat(all_pi_values), epoch)

    return avg_ce_loss, avg_kl_loss, train_acc, mean_pi_i, pi_bar_ema
