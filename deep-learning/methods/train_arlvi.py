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
import math, time


# ---------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------
def tb_log_arlvi(
        writer,
        epoch: int,
        *,
        ce_loss: float,
        kl_loss: float,
        entropy_reg: float,
        train_acc: float,
        pi_mean: float,
        pi_hist: torch.Tensor,           # 1-D tensor of all πᵢ this epoch
        pi_class_means: Dict[int, float] # {class_id: mean π}
):
    """Write metrics to TensorBoard."""
    # ---- losses ------------------------------------------------------
    writer.add_scalars("Loss", {
        "CE_weighted": ce_loss,
        "KL":          kl_loss,
        "EntropyReg":  entropy_reg,
    }, epoch)

    # ---- accuracy ----------------------------------------------------
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    # ---- πᵢ global stats --------------------------------------------
    writer.add_scalar("Pi/global_mean", pi_mean, epoch)
    writer.add_histogram("Pi/distribution", pi_hist, epoch)

    # ---- πᵢ per-class means (no histograms) --------------------------
    for cls, m in pi_class_means.items():
        writer.add_scalar(f"Pi/class_mean/{cls:03d}", m, epoch)


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
    scaler: torch.cuda.amp.GradScaler | None = None,
    model_features:      torch.nn.Module,
    model_classifier:    torch.nn.Module,
    inference_net:       torch.nn.Module,
    dataloader:          torch.utils.data.DataLoader,
    optimizer:           Dict[str, torch.optim.Optimizer],  # keys: backbone / classifier
    inference_optimizer: torch.optim.Optimizer,
    device:              torch.device,
    epoch:               int,
    lambda_kl:           float = 1.0,
    pi_bar:              float = 0.85,   # scalar prior used during warm-up
    warmup_epochs:       int   = 2,
    alpha:               float = 0.85,   # EMA momentum for per-class priors
    pi_bar_class:        torch.Tensor | None = None,  # shape [num_classes]
    beta:                float = 0.4,
    tau:                 float = 1,  # temperature for πᵢ attachment on CE term
    max_gamma:           float = 1,  # max attachment of πᵢ on the CE term (after ramp-up) is full-attachment for analytical optimality
    scheduler:           Dict[str, torch.optim.lr_scheduler._LRScheduler] | None = None,
    writer=None,
    grad_clip:           float = 2.0,
) -> Tuple[float, float, float, torch.Tensor]:

    """Train A-RLVI for **one epoch**.

    Returns
    -------
    avg_ce_loss, avg_kl_loss, train_acc, updated_pi_bar_class
    """

    # Put all sub-nets into training mode
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # Epoch-level book-keeping
    total_loss = total_ce = total_kl = 0.0
    entropy_reg_epoch = 0.0
    total_seen = total_correct = 0
    all_pi_values, all_labels = [], []   # for histograms / EMA update
    eps = 1e-8
    # running sums of per-batch gradient norms (so we can print epoch averages)
    grad_sum_bbk = grad_sum_cls = grad_sum_inf = 0.0      

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
        with torch.cuda.amp.autocast(enabled=scaler is not None):     
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
        pi_i   = (scale * pi_raw + offset).clamp(0.1, 0.9)      # pi_raw ∈ (0,1) → pi_i ∈ (0.05,0.95) --> clamp to ensure no examples get completely ignored


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
        # max_gamma = 1, full attachment of πᵢ on CE after ramp-up
        gamma_ramp_epochs = 5
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
        decay_start, decay_len = 4, 10
        beta_floor = 0.2 
        beta_now = beta * max(beta_floor, 1 - max(epoch - decay_start, 0) / decay_len) # anneal β from beta to beta_floor (0.4 → 0.05)
        entropy_reg = beta_now * (-(pi_i * torch.log(pi_i + eps) +
                                    (1 - pi_i) * torch.log(1 - pi_i + eps))).mean()
        entropy_reg_epoch += entropy_reg.item() * B


        # Rubber-band λ_KL schedule (2.0 → λ over 15 epochs after warm-up)
        decay_rate   = (1.5 - lambda_kl) / 40
        kl_lambda    = 2 - decay_rate * max(epoch - warmup_epochs, 0)
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

        # ----- backward ----------------------------------------------------
        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
        else:
            total_batch_loss.backward()

        # ----- unscale before clipping ------------------------------------
        if scaler is not None:
            scaler.unscale_(optim_bbk)
            scaler.unscale_(optim_cls)
            if epoch >= warmup_epochs:
                scaler.unscale_(inference_optimizer)

        grad_bbk = torch.nn.utils.clip_grad_norm_(model_features.parameters(),   5.0).item()
        grad_cls = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), 5.0).item()
        grad_inf = (torch.nn.utils.clip_grad_norm_(inference_net.parameters(), grad_clip).item()
                    if epoch >= warmup_epochs else float('nan'))
        grad_sum_bbk += grad_bbk * B
        grad_sum_cls += grad_cls * B
        if not math.isnan(grad_inf):
            grad_sum_inf += grad_inf * B


        # ----- optimiser step ---------------------------------------------
        if scaler is not None:
            scaler.step(optim_bbk)
            if epoch >= warmup_epochs:
                scaler.step(optim_cls)
                scaler.step(inference_optimizer)
            scaler.update()
        else:
            optim_bbk.step()
            if epoch >= warmup_epochs:
                optim_cls.step()
                inference_optimizer.step()

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

    # ------------------------------------------------------------------
    avg_ce_loss = total_ce / total_seen
    avg_kl_loss = total_kl / total_seen
    train_acc   = total_correct / total_seen
    avg_entropy_reg = entropy_reg_epoch / total_seen

    # ------------------------------------------------------------------
    # Build global + per-class π statistics for logging
    # ------------------------------------------------------------------
    all_pi_cat     = torch.cat(all_pi_values)   # shape [N]
    all_labels_cat = torch.cat(all_labels)      # shape [N]

    pi_mean_global = all_pi_cat.mean().item()

    pi_class_means: Dict[int, float] = {}
    if pi_bar_class is not None:
        for cls in range(pi_bar_class.size(0)):
            mask = (all_labels_cat == cls)
            if mask.any():
                pi_class_means[cls] = all_pi_cat[mask].mean().item()



    # ───────────────────────────────────────────────────────────────────────
    # Compact console print – one line per epoch (easy hyper-param tuning)
    # ───────────────────────────────────────────────────────────────────────
    lr_bbk = optimizer['backbone'].param_groups[0]['lr']
    lr_cls = optimizer['classifier'].param_groups[0]['lr']

    kl_lambda = max(lambda_kl,
                    3.0 - (3.0 - lambda_kl) / 15 * max(epoch - warmup_epochs, 0))
    gamma = 0.0 if epoch < warmup_epochs else min(max_gamma,
            (epoch - warmup_epochs + 1) / 5 * max_gamma)

    pi_p10, pi_p50, pi_p90 = torch.quantile(all_pi_cat,
                                            torch.tensor([.10, .50, .90])).tolist()

    grad_bbk_epoch = grad_sum_bbk / total_seen
    grad_cls_epoch = grad_sum_cls / total_seen
    grad_inf_epoch = grad_sum_inf / total_seen if grad_sum_inf > 0 else float("nan")

    print(
        f"[ep {epoch:03d}] "
        f"train_acc {train_acc:5.1f}% │ "
        f"CE {avg_ce_loss:.4f} KL {avg_kl_loss:.4f} Ent {avg_entropy_reg:.4f} │ "
        f"γ {gamma:.2f} λ_KL {kl_lambda:.2f} │ "
        f"LR bbk {lr_bbk:.2e} cls {lr_cls:.2e} │ "
        f"|∇| bbk {grad_bbk_epoch:.2f} cls {grad_cls_epoch:.2f} inf {grad_inf_epoch:.2f} │ "
        f"πᵢ μ {pi_mean_global:.2f} p10/p50/p90 {pi_p10:.2f}/{pi_p50:.2f}/{pi_p90:.2f}"
    )



    # ------------------------------------------------------------------
    # TensorBoard logging  (if writer was passed in)
    # ------------------------------------------------------------------
    if writer is not None:
        tb_log_arlvi(
            writer, epoch,
            ce_loss       = avg_ce_loss,
            kl_loss       = avg_kl_loss,
            entropy_reg   = avg_entropy_reg,      # epoch-average value
            train_acc     = train_acc,
            pi_mean       = pi_mean_global,
            pi_hist       = all_pi_cat,
            pi_class_means= pi_class_means,
        )
        writer.flush()

    return avg_ce_loss, avg_kl_loss, train_acc, pi_bar_class
