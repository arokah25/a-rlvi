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
# One-knob, bimodal, non-collapsing A-RLVI (single epoch)
# -----------------------------------------------------------------------------
def train_arlvi_zscore(
    *,
    model_features:   torch.nn.Module,
    model_classifier: torch.nn.Module,
    inference_net:    torch.nn.Module,
    dataloader:       torch.utils.data.DataLoader,
    optim_all:        torch.optim.Optimizer,
    device:           torch.device,
    epoch:            int,
    tau:              float = 0.5,
    ema_alpha:        float = 0.95,
    scaler:           'torch.cuda.amp.GradScaler|None' = None,
    return_diag:      bool  = False,
    log_every:        int   = 200,
    scheduler:        'torch.optim.lr_scheduler._LRScheduler|None' = None,
    grad_clip:        float | None = None,
    pi_bins:          tuple[float, float] = (0.25, 0.75),
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

    # (No τ annealing; τ is fixed by the function argument.)

    # Running stats over the epoch (for averages)
    ce_sum = kl_sum = 0.0            # sums of CE and KL terms (weighted as defined below)
    n_seen = 0                        # total samples seen
    n_correct = 0                     # correct predictions (for accuracy)

    # Grad norm diagnostics (averaged over batches)
    grad_bb_sum  = 0.0                # ∥grad∥ over backbone params per batch
    grad_cls_sum = 0.0                # ∥grad∥ over classifier params per batch
    grad_inf_sum = 0.0                # ∥grad∥ over inference net params per (update) batch

    # --- persistent counters for how many times we snapshot grads ---
    grad_snapshots = 0
    inf_snapshots  = 0  # (kept for compatibility with diagnostics)

    # Keep a small sample of π values for histogram/diagnostics (memory-safe)
    pi_samples, pi_cap = [], 20_000

    # --- diagnostics (epoch aggregates) ---
    spearman_sum = 0.0    # average Spearman(π, -CE)
    spearman_cnt = 0
    pct_low_sum  = 0.0    # % of π < 0.25
    pct_high_sum = 0.0    # % of π > 0.75
    pct_cnt      = 0
    
    # LR traces (per-batch)
    lr_trace_bb:  list[float] = []
    lr_trace_cls: list[float] = []

    # π→correctness accumulators
    bin_lo, bin_hi = pi_bins
    pi_bin_totals  = [0, 0, 0]   # [<lo, lo–hi, >hi]
    pi_bin_correct = [0, 0, 0]

    # Slow global prior π̄ (scalar) maintained as an EMA of mean(π) per batch
    pi_bar_running: torch.Tensor | None = None  # will hold a 0-dim tensor on device

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
        # keep backbone+head under AMP for speed
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            # 1) Backbone → features
            z_i = model_features(images)            # (B, D, 1, 1) for ResNet-like
            z_i = z_i.view(B, -1)                   # (B, D)  flatten spatial dims

            # 2) Classifier → logits over classes
            logits = model_classifier(z_i)          # (B, C) logits for C classes

            # 4) Per-sample supervised loss (e.g., CE) — NO reduction
            ce_vec = F.cross_entropy(logits, labels, reduction="none")  # (B,)

        # 5) DETACHED per-sample target via batch z-scored CE (in FP32):
        #    r_i := zscore(CE) detached so gradients do NOT flow into θ or φ from here
        r_i = _zscore_detached(ce_vec.float())      # (B,) detached FP32

        #    Calibrated teacher with β = logit(π̄_prior):
        #    use the EMA prior from previous batches; initialize at 0.75 on first use
        if pi_bar_running is None:
            pi_bar = torch.tensor(0.75, device=device, dtype=torch.float32)
        else:
            pi_bar = pi_bar_running.detach().float()
        beta = torch.logit(torch.clamp(pi_bar, 1e-4, 1 - 1e-4))  # scalar
        #    q_i(τ) := σ( - r_i / τ + β )  → lower-than-avg CE (easy) → q ≈ 1 (clean)
        q_i = torch.sigmoid(-r_i / tau + beta)     # (B,) detached path overall

        # 3) Inference net → corruption belief π_i  (run in full precision due to LayerNorm)
        with torch.autocast(device_type="cuda", enabled=False):
            pi_i = inference_net(z_i.detach().float())  # (B,)

        # 6) Slow global prior π̄  (scalar) via EMA of mean(π) per batch
        batch_mean_pi = pi_i.mean().detach()    # scalar (0-dim tensor), detached
        # --- update EMA under no_grad so it's a pure statistic ---
        with torch.no_grad():
            if pi_bar_running is None:
                pi_bar_running = batch_mean_pi  # initialize EMA on first batch
            else:
                pi_bar_running = ema_alpha * pi_bar_running + (1. - ema_alpha) * batch_mean_pi

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
        #   Pull π_i toward its detached per-sample target q_i(τ).
        #   (β is baked into q_i, so we drop the extra KL-to-prior.)
        kl_to_q   = kl_bern(pi_i, q_i)                           # (B,)
        L_phi     = kl_to_q.mean()                                # scalar

        #   For logging epoch-averages, we sum CE and KL *as used*:
        #   - CE tracked with DETACHED π weights (matches L_theta use)
        #   - KL tracked as the teacher term only
        ce_term = (pi_w * ce_vec).sum()                           # scalar
        kl_term = kl_to_q.sum()                                   # scalar

        # ----------------- Backward / Optimizer steps -----------------
        # Unified optimizer path: zero grads once, backprop once on (L_theta + L_phi)
        optim_all.zero_grad(set_to_none=True)

        if scaler is not None and torch.cuda.is_available():
            # backward with scaling
            scaler.scale(L_theta + L_phi).backward()

            # ---- UN-SCALE ONCE before any clipping or logging ----
            scaler.unscale_(optim_all)

            # optional clipping (on unscaled grads)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), grad_clip)
        else:
            (L_theta + L_phi).backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), grad_clip)

        # ----- measure grad norms on UN-SCALED grads, BEFORE .step() -----
        if (b_idx + 1) % log_every == 0:
            # measure without modifying grads (max_norm=inf is a no-op)
            total_norm_bb  = torch.nn.utils.clip_grad_norm_(model_features.parameters(),  float('inf'))
            total_norm_cls = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), float('inf'))
            total_norm_inf = torch.nn.utils.clip_grad_norm_(inference_net.parameters(),   float('inf'))
            grad_bb_sum  += float(total_norm_bb)
            grad_cls_sum += float(total_norm_cls)
            grad_inf_sum += float(total_norm_inf)
            grad_snapshots += 1
            inf_snapshots  += 1

            # Lightweight batch log
            tqdm.write(
                f"  ↳ bt {b_idx:04d}: "
                f"Lθ={L_theta.item():.3f} Lφ={L_phi.item():.3f} | "
                f"CĒ={ce_vec.mean().item():.3f}  "
                f"KLq̄={kl_to_q.mean().item():.3f} | "
                f"π̄_batch={float(batch_mean_pi):.3f}  π̄_ema={float(pi_bar_running):.3f} | "
                f"π_min={pi_i.min().item():.3f} π_max={pi_i.max().item():.3f} | "
                f"spearman(pi_vs_negCE)={_spearman_corr_torch(pi_i, -ce_vec):.3f}  "
                f"pct_pi_below_0.25={(pi_i < 0.25).float().mean().item()*100:.1f}%  "
                f"pct_pi_above_0.75={(pi_i > 0.75).float().mean().item()*100:.1f}%"
            )

        # ----- now do optimizer.step() / scaler.step() -----
        if scaler is not None and torch.cuda.is_available():
            scaler.step(optim_all)
            scaler.update()
        else:
            optim_all.step()

        # Step LR schedulers *per batch* (after optimizer.step)
        if scheduler is not None:
            # capture current LRs used for THIS batch before stepping scheduler
            try:
                lr_trace_bb.append(optim_all.param_groups[0]['lr'])  # backbone group
                lr_trace_cls.append(optim_all.param_groups[2]['lr']) # head group
            except Exception:
                pass
            scheduler.step()

        # ----------------- Stats / Diagnostics -----------------
        ce_sum += ce_term.item()
        kl_sum += kl_term.item()
        n_seen += B

        # Accuracy: argmax over logits vs labels
        n_correct += logits.argmax(1).eq(labels).sum().item()

        # Keep a small sample of π values for histogram later
        ps = pi_i.detach().flatten().cpu()
        if ps.numel() and sum(t.numel() for t in pi_samples) < pi_cap:
            pi_samples.append(ps[: min(1024, ps.numel())])

        # --- additional per-batch diagnostics (epoch aggregates) ---
        rho = _spearman_corr_torch(pi_i, -ce_vec)
        spearman_sum += rho
        spearman_cnt += 1

        pct_low  = (pi_i < 0.25).float().mean().item()
        pct_high = (pi_i > 0.75).float().mean().item()
        pct_low_sum  += pct_low
        pct_high_sum += pct_high
        pct_cnt      += 1

        # π→correctness accumulation (per batch)
        with torch.no_grad():
            preds = logits.argmax(1)
            correct_mask = preds.eq(labels)
            p = pi_i

            m0 = p < bin_lo
            m1 = (p >= bin_lo) & (p <= bin_hi)
            m2 = p > bin_hi

            pi_bin_totals[0]  += int(m0.sum().item())
            pi_bin_totals[1]  += int(m1.sum().item())
            pi_bin_totals[2]  += int(m2.sum().item())

            if m0.any():
                pi_bin_correct[0] += int(correct_mask[m0].sum().item())
            if m1.any():
                pi_bin_correct[1] += int(correct_mask[m1].sum().item())
            if m2.any():
                pi_bin_correct[2] += int(correct_mask[m2].sum().item())


    # ----------------- Once-per-epoch φ step (if chosen) -----------------
    # (Removed: unified optimizer updates φ per-batch; no extra epoch-level step.)

    # ----------------- Epoch averages -----------------
    ce_epoch  = ce_sum / n_seen
    kl_epoch  = kl_sum / n_seen
    acc_epoch = n_correct / n_seen

    # --- use the snapshot counters for averaging ---
    grad_bb  = grad_bb_sum  / max(1, grad_snapshots)
    grad_cls = grad_cls_sum / max(1, grad_snapshots)
    grad_inf = grad_inf_sum / max(1, inf_snapshots)

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
        # finalize π→correctness bins
    def _safe_div(a: int, b: int) -> float:
        return (a / b) if b > 0 else 0.0

    pi_acc_bins = {
        "lt_0.25":  _safe_div(pi_bin_correct[0], pi_bin_totals[0]),
        "0.25_0.75": _safe_div(pi_bin_correct[1], pi_bin_totals[1]),
        "gt_0.75":  _safe_div(pi_bin_correct[2], pi_bin_totals[2]),
    }
    pi_bin_counts = {
        "lt_0.25":  int(pi_bin_totals[0]),
        "0.25_0.75": int(pi_bin_totals[1]),
        "gt_0.75":  int(pi_bin_totals[2]),
    }

    diag = {
        "grad_backbone":   grad_bb,
        "grad_classifier": grad_cls,
        "grad_inference":  grad_inf,
        **pi_diag,
        "pi_bar": float(pi_bar_running) if pi_bar_running is not None else 0.0,
        # extras
        "spearman_rank_pi_vs_negCE": float(rho_epoch),     # Rank correlation between π and -CE
        "percent_pi_below_0.25":      float(pct_low_epoch), # % samples with π < 0.25 (likely corrupt)
        "percent_pi_above_0.75":      float(pct_high_epoch) # % samples with π > 0.75 (likely clean)
    }

    # attach LR traces and π→correctness to diag
    diag.update({
        "lr_trace_backbone":   lr_trace_bb,
        "lr_trace_classifier": lr_trace_cls,
        "pi_acc_bins":         pi_acc_bins,
        "pi_bin_counts":       pi_bin_counts,
        "pi_bins":             (bin_lo, bin_hi),
    })

    if return_diag:
        return ce_epoch, kl_epoch, acc_epoch, diag
    else:
        return ce_epoch, kl_epoch, acc_epoch
# -----------------------------------------------------------------------------
