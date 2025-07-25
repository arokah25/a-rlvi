"""
train_arlvi.py
==============

One-epoch training routine for **A-RLVI** with full stability tricks **and**
extensive diagnostics printed every 500 batches.

Key implementation notes
------------------------
1. **Soft squashing of πᵢ** keeps gradients alive in (0.05, 0.95).
2. **Rubber-band KL weight** linearly decays 3.0 → λ over 15 epochs (after warm-up).
3. **Class-conditioned prior π̄_c** scalar during warm-up, per-class EMA afterwards.
4. **Detach πᵢ from CE** avoids feedback loop between classifier errors and πᵢ.
5. **Entropy regularisation** β linearly decays 0.4 → 0 between epochs 12-22.
6. **Gradient clipping** 5.0 on backbone/classifier, configurable on inference net.
7. **Diagnostics** entropy, prior range, grad norms, LR, cleanest/noisiest class.

This file is *stand-alone* – no external globals except the `pi_bar_class` tensor
passed in by caller.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Helper – closed-form KL divergence between two Bernoulli distributions
# -----------------------------------------------------------------------------

def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """KL( Bern(πᵢ) ‖ Bern(π̄) ) – works element-wise for any matching shapes."""
    eps = 1e-6                           # avoid log(0)
    pi_i   = pi_i.clamp(eps, 1. - eps)
    pi_bar = pi_bar.clamp(eps, 1. - eps)
    return pi_i * torch.log(pi_i / pi_bar) + (1. - pi_i) * torch.log((1. - pi_i) / (1. - pi_bar))


# -----------------------------------------------------------------------------
# Main training routine (one epoch)
# -----------------------------------------------------------------------------

def train_arlvi(
    *,
    model_features:      torch.nn.Module,
    model_classifier:    torch.nn.Module,
    inference_net:       torch.nn.Module,
    dataloader:          torch.utils.data.DataLoader,
    optimizer:           Dict[str, torch.optim.Optimizer],  # keys: backbone / classifier
    inference_optimizer: torch.optim.Optimizer,
    device:              torch.device,
    epoch:               int,
    lambda_kl:           float = 1.0,
    pi_bar:              float = 0.75,   # scalar prior used during warm-up
    warmup_epochs:       int   = 2,
    alpha:               float = 0.85,   # EMA momentum for per-class priors
    pi_bar_class:        torch.Tensor | None = None,  # shape [num_classes]
    beta:                float = 0.4,
    tau:                 float = 0.6,
    max_gamma:           float = 0.2,  # max attachment of πᵢ on the CE term (after ramp-up)
    scheduler:           Dict[str, torch.optim.lr_scheduler._LRScheduler] | None = None,
    writer=None,
    grad_clip:           float = 2.0,
) -> Tuple[float, float, float, torch.Tensor, Dict[str, torch.Tensor]]:
    """Train A-RLVI for **one epoch**.

    Returns
    -------
    avg_ce_loss, avg_kl_loss, train_acc, updated_pi_bar_class, histogram_data
    """

    # Put all sub-nets into training mode
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # Epoch-level book-keeping
    total_loss = total_ce = total_kl = 0.0
    total_seen = total_correct = 0
    all_pi_values, all_labels = [], []   # for histograms / EMA update
    eps = 1e-8

    # ------------------------------------------------------------------
    # Mini-batch loop
    # ------------------------------------------------------------------
    for batch_idx, (images, labels, *_) in enumerate(dataloader):
        # Move batch to device
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        B = images.size(0) # batch size

        # --------------------------------------------------------------
        # Forward pass – feature extractor + (optionally frozen) classifier
        # --------------------------------------------------------------
        # model_features: backbone (e.g., ResNet50) extracts features zᵢ
        # model_classifier: classifier (e.g., linear layer) predicts logits
        # z_i: shape [B, d] where d is the feature dimension (e.g., 2048 for ResNet50)
        # logits: shape [B, num_classes] – class predictions
        z_i = model_features(images).view(B, -1) # [B, d]
        logits = model_classifier(z_i) # [B, num_classes]

        # --------------------------------------------------------------
        # Inference network predicts πᵢ; gradients flow only after warm-up
        # --------------------------------------------------------------
        with torch.set_grad_enabled(epoch >= warmup_epochs):
            pi_raw = inference_net(z_i)           # unconstrained in (0,1)

        # Soft squashing so πᵢ ∈ (0.05,0.95) and ramp-up keeps early grads alive
        #ramp_T = 6
        #t      = min(epoch, ramp_T) / ramp_T      # 0 → 1 across first 6 epochs
        #scale  = 0.40 + 0.55 * t                 # 0.40 → 0.95
        #offset = 0.5 - scale / 2
        #pi_i   = scale * pi_raw + offset          # shape [B,1] or [B]

        # Soft squashing of πᵢ 
        scale  = 0.90                         # width of the open interval
        offset = 0.05                         # left border
        pi_i   = scale * pi_raw + offset      # pi_raw ∈ (0,1) → pi_i ∈ (0.05,0.95)

        # Save for histograms of pi_i and EMA updates of per-class priors
        all_pi_values.append(pi_i.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # --------------------------------------------------------------
        # Class-conditioned prior π̄_c
        #   • scalar during warm-up;  per-class tensor afterwards
        # --------------------------------------------------------------
        if epoch < warmup_epochs or pi_bar_class is None:
            prior_value = torch.full_like(pi_i, pi_bar)          # [B,1] scalar prior
        else:
            # pi_bar_class: shape [num_classes]
            # labels: shape [B] with class indices
            # prior_value: shape [B,1] with prior for each sample
            # For each element in labels, PyTorch looks up the 
            # corresponding entry in pi_bar_class.
            # Result: a 1-D tensor of length B whose i-th value is 
            # the prior for that sample’s true class
            prior_value = pi_bar_class[labels].unsqueeze(1)      # [B,1]

        # --------------------------------------------------------------
        # Loss components
        # --------------------------------------------------------------
        ce_loss   = F.cross_entropy(logits, labels, reduction='none')         # [B]
        kl_loss   = compute_kl_divergence(pi_i, prior_value)                  # [B,1]
        mean_kl   = kl_loss.mean()
        pi_temp   = pi_i ** tau 

        # Partial detachment of πᵢ on the CE term with ramp-up
        gamma_ramp_epochs = 1
        if epoch < warmup_epochs:
            gamma = 0.0  # full detachment during warm-up
        else:
            # linear ramp to gamma_max over gamma_ramp epochs
            progress = (epoch - warmup_epochs + 1) / gamma_ramp_epochs
            gamma = min(max_gamma, progress * max_gamma)
        

        # -- compute CE weight -----
        # element-wise weights
        ce_weight = (1 - gamma) * pi_temp.detach() + gamma * pi_temp   # blend
        weighted_ce_loss = (ce_weight * ce_loss).mean()  

        # Entropy regulariser – linearly annealed β
        decay_start, decay_len = 4, 14
        beta_now = beta * max(0.1, 1 - max(epoch - decay_start, 0) / decay_len)
        entropy_reg = beta_now * (-(pi_i * torch.log(pi_i + eps) +
                                    (1 - pi_i) * torch.log(1 - pi_i + eps))).mean()

        # Rubber-band λ_KL schedule (2.0 → λ over 15 epochs after warm-up)
        decay_rate   = (2.0 - lambda_kl) / 15.0
        kl_lambda    = 2.0 - decay_rate * max(epoch - warmup_epochs, 0)
        kl_lambda    = max(lambda_kl, kl_lambda)

        total_batch_loss = weighted_ce_loss + kl_lambda * mean_kl - entropy_reg

        # --------------------------------------------------------------
        # Back-prop and optimiser / scheduler steps
        # --------------------------------------------------------------
        optim_bbk = optimizer['backbone']
        optim_cls = optimizer['classifier']
        
        optim_bbk.zero_grad(set_to_none=True)
        optim_cls.zero_grad(set_to_none=True)
        if epoch >= warmup_epochs:
            inference_optimizer.zero_grad(set_to_none=True)

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_features.parameters(),   5.0)
        torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), 5.0)
        if epoch >= warmup_epochs:
            torch.nn.utils.clip_grad_norm_(inference_net.parameters(), grad_clip)

        # Optimisers step
        # Note: inference and classifier optimizers are only updated after warm-up
        optim_bbk.step()
        if epoch >= warmup_epochs:
            optim_cls.step()
            inference_optimizer.step()

        # Scheduler step after optimiser step (PyTorch recommendation)
        if scheduler is not None:
            scheduler['backbone'].step()
            scheduler['classifier'].step()

        # --------------------------------------------------------------
        # Running aggregates
        # --------------------------------------------------------------
        total_loss   += total_batch_loss.item() * B
        total_ce     += weighted_ce_loss.item() * B
        total_kl     += mean_kl.item() * B
        total_seen   += B
        total_correct += logits.argmax(1).eq(labels).sum().item()

        # --------------------------------------------------------------
        # Diagnostics every 500 batches
        # --------------------------------------------------------------
        if batch_idx % 500 == 0:
            # --- entropy of πᵢ in current batch
            entropy_val = (-(pi_i * torch.log(pi_i + eps) + (1 - pi_i) * torch.log(1 - pi_i + eps))).mean().item()

            # --- prior range
            if pi_bar_class is not None:
                pi_bar_min = pi_bar_class.min().item()
                pi_bar_max = pi_bar_class.max().item()
            else:
                pi_bar_min = pi_bar_max = pi_bar

            # --- identify cleanest / noisiest class in *this* batch
            with torch.no_grad():
                batch_means = {}
                for cls in labels.unique():
                    cls_mask = labels == cls
                    batch_means[int(cls)] = pi_i[cls_mask].mean().item()
                sorted_batch = sorted(batch_means.items(), key=lambda kv: kv[1])
                top_noisy, top_clean = sorted_batch[0][0], sorted_batch[-1][0]

            # --- gradient norms + LRs
            grad_inf = torch.nn.utils.clip_grad_norm_(inference_net.parameters(), float('inf')).item()
            grad_cls = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), float('inf')).item()
            grad_bbk = torch.nn.utils.clip_grad_norm_(model_features.parameters(), float('inf')).item()
            lr_bbk   = optim_bbk.param_groups[0]['lr']
            lr_cls   = optim_cls.param_groups[0]['lr']

            # Optional TensorBoard scalar logging per batch
            if writer is not None:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('LR/backbone', lr_bbk, step)
                writer.add_scalar('LR/classifier', lr_cls, step)
                writer.add_scalar('GradNorm/Inference', grad_inf, step)
                writer.add_scalar('GradNorm/Classifier', grad_cls, step)

            # Console debug print
            print(
                f"[Ep {epoch:02d} Bt {batch_idx:04d}] "
                f"kl_loss={mean_kl.item():.3f}, ce_loss={weighted_ce_loss.item():.3f} "
                f"πᵢ batch_μ={pi_i.mean():.3f} batch_min={pi_i.min():.2f} batch_max={pi_i.max():.2f} "
                f"batch_entropy={entropy_val:.2f} "
                f"pī_c∈[{pi_bar_min:.2f},{pi_bar_max:.2f}] "
                f"cleanest_cls={top_clean:03d} noisiest_cls={top_noisy:03d} "
                f"|∇φ|={grad_inf:.2f} |∇θcls|={grad_cls:.2f} |∇θbbk|={grad_bbk:.2f} "
                f"lr_cls={lr_cls:.6f} lr_bbk={lr_bbk:.6f}")

    # ------------------------------------------------------------------
    # End-of-epoch updates – EMA for class priors + scalar prior update
    # ------------------------------------------------------------------
    if pi_bar_class is not None and epoch >= warmup_epochs:
        #concatenate all pi_i values and labels in the epoch for class-wise updates
        all_pi_cat     = torch.cat(all_pi_values)
        all_labels_cat = torch.cat(all_labels)
        for cls in range(pi_bar_class.size(0)):
            mask = (all_labels_cat == cls) # "mask": Boolean tensor that selects all samples of class 'cls' in this epoch
            if mask.any():
                # Exponential moving average update for class priors
                pi_cls_mean = all_pi_cat[mask].mean().to(device)
                pi_bar_class[cls] = alpha * pi_bar_class[cls] + (1 - alpha) * pi_cls_mean

    mean_pi_i = torch.cat(all_pi_values).mean().item() # overall mean πᵢ in epoch

    avg_ce_loss = total_ce / total_seen
    avg_kl_loss = total_kl / total_seen
    train_acc   = total_correct / total_seen

    # TensorBoard epoch-level scalars / histograms
    if writer is not None:
        writer.add_scalar('Loss/CE_weighted', avg_ce_loss, epoch)
        writer.add_scalar('Loss/KL',          avg_kl_loss, epoch)
        writer.add_scalar('Inference/pi_i_mean', mean_pi_i, epoch)
        if epoch % 3 == 0:
            writer.add_histogram(f'Inference/PiDistribution epoch {epoch}', torch.cat(all_pi_values), epoch)

    # Histogram data  – only keep for cleanest/noisiest classes for caller plotting
    hist_data: Dict[str, torch.Tensor] = {}
    if epoch % 3 == 0 and pi_bar_class is not None:
        cls_order = pi_bar_class.detach().cpu().numpy().argsort()
        cleanest  = cls_order[-5:]
        noisiest  = cls_order[:5]
        all_pi_cat     = torch.cat(all_pi_values)
        all_labels_cat = torch.cat(all_labels)
        for tag, cls_ids in [('cleanest', cleanest), ('noisiest', noisiest)]:
            for c in cls_ids:
                mask = (all_labels_cat == c)
                if mask.any():
                    hist_data[f'{tag}_cls_{c}'] = all_pi_cat[mask]

    return avg_ce_loss, avg_kl_loss, train_acc, pi_bar_class, hist_data
