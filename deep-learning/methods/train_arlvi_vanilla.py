
from __future__ import annotations  # postpone evaluation of type hints (safer)
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility: KL( Bern(π) ‖ Bern(π̄) ) element-wise
# -----------------------------------------------------------------------------
def kl_bern(pi: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """
    KL(Bern(pi) || Bern(pi_bar)), computed element-wise.
    Shapes:
      - pi:     (B,)
      - pi_bar: (B,) or scalar broadcastable to (B,)
    Returns:
      - kl:     (B,)
    """
    eps = 1e-6
    pi     = pi.clamp(eps, 1. - eps)
    pi_bar = pi_bar.clamp(eps, 1. - eps)
    return pi * (pi.log() - pi_bar.log()) + (1. - pi) * ((1. - pi).log() - (1. - pi_bar).log())


# -----------------------------------------------------------------------------
# Utility: Batch z-score of a vector, fully DETACHED from autograd
# -----------------------------------------------------------------------------
def _zscore_detached(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batch-wise z-score of x with gradient DETACHED.
    Shapes:
      - x: (B,)  (e.g., per-sample cross-entropy)
    Returns:
      - z: (B,) detached, zero-mean, unit-ish variance (numerically safe)
    Notes:
      - Uses unbiased=False so std is computed as sqrt(E[(x-μ)^2]) over the batch.
      - Adds eps to avoid division by ~0 in very uniform batches.
      - The .detach() is CRITICAL: inference net should NOT receive gradients through CE.
    """
    mu = x.mean()                     # scalar
    sd = x.std(unbiased=False)        # scalar
    z  = (x - mu) / (sd + eps)        # (B,)
    return z.detach()                 # drop gradients


# -----------------------------------------------------------------------------
# Utility: Spearman correlation ρ( a , b ) computed on tensors (detached)
# -----------------------------------------------------------------------------
def _spearman_corr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute Spearman rank correlation between 1D tensors a and b on the same device.
    Returns a Python float. If either vector is constant, returns 0.0.
    """
    a = a.detach().reshape(-1)
    b = b.detach().reshape(-1)
    n = a.numel()
    if n < 2:
        return 0.0

    # ranks via argsort twice
    ra_idx = torch.argsort(a)
    rb_idx = torch.argsort(b)
    ra = torch.empty_like(ra_idx, dtype=torch.float32)
    rb = torch.empty_like(rb_idx, dtype=torch.float32)
    ra[ra_idx] = torch.arange(n, device=a.device, dtype=torch.float32)
    rb[rb_idx] = torch.arange(n, device=b.device, dtype=torch.float32)

    ra = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-12)
    rb = (rb - rb.mean()) / (rb.std(unbiased=False) + 1e-12)
    denom = (ra.std(unbiased=False) * rb.std(unbiased=False)).item()
    if denom == 0.0 or torch.isnan(ra).any() or torch.isnan(rb).any():
        return 0.0
    rho = (ra * rb).mean().item()
    return float(rho)


# -----------------------------------------------------------------------------
# One-knob, bimodal, non-collapsing A-RLVI (single epoch, HEAVILY COMMENTED)
# -----------------------------------------------------------------------------
def train_arlvi_vanilla(
    *,
    model_features:   torch.nn.Module,          # backbone   θ_bb
    model_classifier: torch.nn.Module,          # classifier θ_cls
    inference_net:    torch.nn.Module,          # f_φ: z -> π
    dataloader:       torch.utils.data.DataLoader,
    optim_backbone:   torch.optim.Optimizer,
    optim_classifier: torch.optim.Optimizer,
    optim_inference:  torch.optim.Optimizer,
    device:           torch.device,
    epoch:            int,

    # === The ONE knob ===
    tau:              float = 1.0,              # temperature for q_i(τ)

    # === Stability knobs (fixed constants; not meant to be tuned) ===
    update_inference_every: str = "batch",      # 'batch' | 'epoch'
    ema_alpha:        float = 0.95,             # EMA coefficient for global prior π̄

    # === (Optional) Mixed precision scaler ===
    scaler:           'torch.cuda.amp.GradScaler|None' = None,  # if provided, used as explained below

    # === Diagnostics / logging ===
    return_diag:      bool  = False,
    log_every:        int   = 200,
):
    """
    Trains A-RLVI for ONE epoch with:
      - π-detached CE for θ-updates (breaks collapse loop),
      - φ-target from batch-zscored CE via q_i(τ),
      - EMA global prior π̄ for stability,
      - Only one hyperparameter: tau.

    Returns:
      (CE_avg, KL_avg, acc_avg) or (..., diag) if return_diag=True
    """

    # Put all nets into train mode
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # Running stats over the epoch (for averages)
    ce_sum = kl_sum = 0.0            # sums of CE and KL terms (weighted as defined below)
    n_seen = 0                        # total samples seen
    n_correct = 0                     # correct predictions (for accuracy)

    # Grad norm diagnostics (averaged over batches)
    grad_bb_sum  = 0.0                # ∥grad∥ over backbone params per batch
    grad_cls_sum = 0.0                # ∥grad∥ over classifier params per batch
    grad_inf_sum = 0.0                # ∥grad∥ over inference net params per (update) batch
    n_inf_batches = 0

    # Keep a small sample of π values for histogram/diagnostics (memory-safe)
    pi_samples, pi_cap = [], 20_000

    # --- NEW: nice-to-have diagnostics (epoch aggregates) ---
    spearman_sum = 0.0    # average Spearman(π, -CE)
    spearman_cnt = 0
    pct_low_sum  = 0.0    # % of π < 0.2
    pct_high_sum = 0.0    # % of π > 0.8
    pct_cnt      = 0

    # Slow global prior π̄ (scalar) maintained as an EMA of mean(π) per batch
    pi_bar_running: torch.Tensor | None = None  # will hold a 0-dim tensor on device

    # If doing once-per-epoch φ update, zero its grads up front
    if update_inference_every == "epoch":
        optim_inference.zero_grad(set_to_none=True)

    # ------------------------------
    # Mini-batch training loop
    # ------------------------------
    for b_idx, (images, labels, *_) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch:03d}")):
        # images: (B, C, H, W)
        # labels: (B,)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B      = images.size(0)

        # ----------------- Forward pass (AMP-safe) -----------------
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            # 1) Backbone → features
            z_i = model_features(images)            # (B, D, 1, 1) for ResNet-like
            z_i = z_i.view(B, -1)                   # (B, D)  flatten spatial dims

            # 2) Classifier → logits over classes
            logits = model_classifier(z_i)          # (B, C) logits for C classes

            # 3) Inference net → corruption belief π_i
            pi_i = inference_net(z_i.detach())      # (B,)

            # Defining inference target q_i(τ) -- 4 and 5 below:
            # 4) Per-sample supervised loss (e.g., CE) — NO reduction
            ce_vec = F.cross_entropy(logits, labels, reduction="none")  # (B,)

            # 5) DETACHED per-sample target via batch z-scored CE:
            #    r_i := zscore(CE) detached so gradients do NOT flow into θ or φ from here
            r_i = _zscore_detached(ce_vec)          # (B,) detached
            #    q_i(τ) := σ( - r_i / τ )  → lower-than-avg CE (easy) → q ≈ 1 (clean)
            #q_i = torch.sigmoid(-r_i / tau)         # (B,) detached (r_i is detached) --- (I) version A non-scaled

            # 6) Slow global prior π̄  (scalar) via EMA of mean(π) per batch
            batch_mean_pi = pi_i.mean().detach()    # scalar (0-dim tensor), detached
            # --- update EMA under no_grad so it's a pure statistic ---
            with torch.no_grad():
                if pi_bar_running is None:
                    pi_bar_running = batch_mean_pi  # initialize EMA on first batch
                else:
                    pi_bar_running = ema_alpha * pi_bar_running + (1. - ema_alpha) * batch_mean_pi
            pi_bar = pi_bar_running.detach()        # scalar, detached

            #####################
            # --- 6b) CALIBRATE q_i so its batch mean matches the EMA prior π̄ ---
            # Start from the uncalibrated target built from z-scored CE (keeps order info)
            eps_q = 1e-6
            q0 = torch.sigmoid(-r_i / tau).clamp(eps_q, 1.0 - eps_q)  # (B,)
            logits0 = torch.logit(q0)                                  # (B,)

            # Target mean is the EMA prior (scalar); keep everything on the same device
            target = float(pi_bar.clamp(eps_q, 1.0 - eps_q))           # Python float is fine here

            # Good initial guess for the bias by matching logits of means
            m0 = float(q0.mean().clamp(eps_q, 1.0 - eps_q))
            b = torch.tensor(
                torch.logit(torch.tensor(target, device=logits0.device))
                - torch.logit(torch.tensor(m0,     device=logits0.device)),
                device=logits0.device
            )

            # 1–2 Newton steps: f(b)=mean(sigmoid(logits0+b)) - target; f'(b)=mean(s*(1-s))
            for _ in range(2):
                s = torch.sigmoid(logits0 + b)      # (B,)
                f = s.mean() - target               # scalar
                g = (s * (1.0 - s)).mean()          # scalar
                b = b - f / (g + 1e-8)

            # Clamp extreme shifts (safety on odd batches)
            b = b.clamp(-10.0, 10.0)

            # Final calibrated target; DETACH so no gradients flow through the target
            q_i = torch.sigmoid(logits0 + b).detach()   # (B,)
            ###################


            # --- normalize θ-weights so their batch mean is exactly 1 ---
            # keeps the overall gradient scale for θ steady across batches.
            # Use the detached π so no gradients flow into φ from the CE path.
            pi_w = pi_i.detach() / (batch_mean_pi + 1e-8)   # shape (B,), mean(pi_w) == 1


            # 7) Loss definitions (one for inf_net and one for the backbone / classifier) 

            #   θ-loss (backbone + classifier):
            #   L_theta = mean( stop_grad(π_i) * CE_i )
            #   - Using stop_grad(π_i) BREAKS the collapse loop:
            #     φ cannot lower L_theta by shrinking π globally.
            L_theta = (pi_w * ce_vec).mean()   # scalar

            #   φ-loss (inference net):
            #   Pull π_i toward its detached per-sample target q_i(τ)
            #   AND anchor it to the slow global prior π̄.
            #   Both inputs to KL are element-wise in (B,).
            kl_to_q   = kl_bern(pi_i, q_i)                           # (B,)
            kl_to_bar = kl_bern(pi_i, torch.full_like(pi_i, float(pi_bar))) # (B,)
            L_phi     = (kl_to_q + kl_to_bar).mean()                 # scalar

            #   For logging epoch-averages, we sum CE and KL *as used*:
            #   - CE tracked with DETACHED π weights (matches L_theta use)
            #   - KL tracked as sum of both KL terms
            ce_term = (pi_w * ce_vec).sum()                          # scalar
            kl_term = (kl_to_q + kl_to_bar).sum()                    # scalar

        # ----------------- Backward / Optimizer steps -----------------
        # Zero grads
        for opt in (optim_backbone, optim_classifier):
            opt.zero_grad(set_to_none=True)
        if update_inference_every == "batch":
            optim_inference.zero_grad(set_to_none=True)

        # --- optional mixed-precision with GradScaler ---
        if scaler is not None and torch.cuda.is_available():
            if update_inference_every == "batch":
                # Batch mode: safe to scale joint loss and step all optimizers now.
                scaler.scale(L_theta + L_phi).backward()
                scaler.step(optim_backbone)
                scaler.step(optim_classifier)
                scaler.step(optim_inference)
                scaler.update()
            else:
                # Epoch mode: θ steps every batch, φ accumulates over the epoch.
                # Scale only L_theta so scaler.update does not affect φ's accumulated grads.
                scaler.scale(L_theta).backward()
                # φ backward unscaled (still under autocast, but not scaled by GradScaler)
                L_phi.backward()
                scaler.step(optim_backbone)
                scaler.step(optim_classifier)
                scaler.update()
        else:
            # No scaler: standard FP32 training
            (L_theta + L_phi).backward()
            optim_backbone.step()
            optim_classifier.step()
            if update_inference_every == "batch":
                optim_inference.step()

        # ----------------- Stats / Diagnostics -----------------
        ce_sum += ce_term.item()
        kl_sum += kl_term.item()
        n_seen += B

        # Accuracy: argmax over logits vs labels
        n_correct += logits.argmax(1).eq(labels).sum().item()

        # Per-batch grad norms (rough, but useful trend indicators)
        grad_bb_batch  = sum((p.grad.norm().item() for p in model_features.parameters()   if p.grad is not None))
        grad_cls_batch = sum((p.grad.norm().item() for p in model_classifier.parameters() if p.grad is not None))
        grad_bb_sum  += grad_bb_batch
        grad_cls_sum += grad_cls_batch

        if update_inference_every == "batch":
            grad_inf_batch = sum((p.grad.norm().item() for p in inference_net.parameters() if p.grad is not None))
            grad_inf_sum   += grad_inf_batch
            n_inf_batches  += 1

        # Keep a small sample of π values for histogram later
        ps = pi_i.detach().flatten().cpu()
        if ps.numel() and sum(t.numel() for t in pi_samples) < pi_cap:
            pi_samples.append(ps[: min(1024, ps.numel())])

        # --- NEW: nice-to-have diagnostics per batch ---
        # Spearman correlation between π and -CE (higher is better separation)
        rho = _spearman_corr_torch(pi_i, -ce_vec)
        spearman_sum += rho
        spearman_cnt += 1

        # Percent of extreme π values (proxy for bimodality)
        pct_low  = (pi_i < 0.2).float().mean().item()
        pct_high = (pi_i > 0.8).float().mean().item()
        pct_low_sum  += pct_low
        pct_high_sum += pct_high
        pct_cnt      += 1

        # Lightweight batch log
        if (b_idx + 1) % log_every == 0:
            tqdm.write(
                f"  ↳ bt {b_idx:04d}: "
                f"Lθ={L_theta.item():.3f} Lφ={L_phi.item():.3f} | "
                f"CĒ={ce_vec.mean().item():.3f}  "
                f"KLq̄={kl_to_q.mean().item():.3f}  KL_π̄={kl_to_bar.mean().item():.3f} | "
                f"π̄_batch={float(batch_mean_pi):.3f}  π̄_ema={float(pi_bar):.3f} | "
                f"π_min={pi_i.min().item():.3f} π_max={pi_i.max().item():.3f} | "
                f"spearman(pi_vs_negCE)={rho:.3f}  pct_pi_below_0.2={pct_low*100:.1f}%  pct_pi_above_0.8={pct_high*100:.1f}%"
            )

    # ----------------- Once-per-epoch φ step (if chosen) -----------------
    if update_inference_every == "epoch":
        # Average φ-gradients across all batches (simple 1/T scale)
        for p in inference_net.parameters():
            if p.grad is not None:
                p.grad.div_(len(dataloader))

        if scaler is not None and torch.cuda.is_available():
            # In epoch mode we accumulated unscaled grads for φ; step without scaler.
            optim_inference.step()
        else:
            optim_inference.step()

        # For diagnostics, grab a final grad norm snapshot
        grad_inf_sum  = sum((p.grad.norm().item() for p in inference_net.parameters() if p.grad is not None))
        n_inf_batches = 1

    # ----------------- Epoch averages -----------------
    ce_epoch  = ce_sum / n_seen
    kl_epoch  = kl_sum / n_seen
    acc_epoch = n_correct / n_seen

    grad_bb  = grad_bb_sum  / len(dataloader)
    grad_cls = grad_cls_sum / len(dataloader)
    grad_inf = grad_inf_sum / max(1, n_inf_batches)

    # Collate π diagnostics
    if pi_samples:
        pi_cat = torch.cat(pi_samples)   # (N_sampled,)
        pi_diag = {
            "pi_min":  float(pi_cat.min()),
            "pi_max":  float(pi_cat.max()),
            "pi_mean": float(pi_cat.mean()),
        }
    else:
        pi_diag = {"pi_min": 0.0, "pi_max": 0.0, "pi_mean": 0.0}

    # --- epoch-level diagnostics ---
    rho_epoch     = spearman_sum / max(1, spearman_cnt)
    pct_low_epoch = (pct_low_sum  / max(1, pct_cnt)) * 100.0
    pct_high_epoch= (pct_high_sum / max(1, pct_cnt)) * 100.0

    diag = {
        "grad_backbone":   grad_bb,
        "grad_classifier": grad_cls,
        "grad_inference":  grad_inf,
        **pi_diag,
        "pi_bar": float(pi_bar_running) if pi_bar_running is not None else 0.0,
        # extras
        "spearman_rank_pi_vs_negCE": float(rho_epoch),     # Rank correlation between π and -CE
        "percent_pi_below_0.2":      float(pct_low_epoch), # % samples with π < 0.2 (likely corrupt)
        "percent_pi_above_0.8":      float(pct_high_epoch) # % samples with π > 0.8 (likely clean)

    }

    if return_diag:
        return ce_epoch, kl_epoch, acc_epoch, diag
    else:
        return ce_epoch, kl_epoch, acc_epoch
