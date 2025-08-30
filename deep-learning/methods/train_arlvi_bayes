# methods/train_arlvi_bayes.py

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------------------------------------------------
# KL( Bern(π) || Bern(π̄) ) element-wise
# NOTE: same utility signature as in zscore code, so main logging stays aligned.
# -------------------------------------------------------------------------
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
    # KL = pi * log(pi/pi_bar) + (1-pi) * log((1-pi)/(1-pi_bar))
    return pi * (pi.log() - pi_bar.log()) + (1. - pi) * ((1. - pi).log() - (1. - pi_bar).log())

# -------------------------------------------------------------------------
# Numerically-stable logit(u) for u ∈ (0,1)
# -------------------------------------------------------------------------
def _logit_clamped(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    log(u/(1-u)) with clamping for numerical safety.
    """
    u = u.clamp(eps, 1. - eps)
    return torch.log(u) - torch.log1p(-u)

# -------------------------------------------------------------------------
# Spearman correlation ρ(a, b) on tensors (detached)
# Same helper as in zscore method so diagnostics are consistent.
# -------------------------------------------------------------------------
def _spearman_corr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1)
    b = b.detach().reshape(-1)
    n = a.numel()
    if n < 2:
        return 0.0
    ra_idx = torch.argsort(a)
    rb_idx = torch.argsort(b)
    ra = torch.empty_like(ra_idx, dtype=torch.float32)
    rb = torch.empty_like(rb_idx, dtype=torch.float32)
    ra[ra_idx] = torch.arange(n, device=a.device, dtype=torch.float32)
    rb[rb_idx] = torch.arange(n, device=b.device, dtype=torch.float32)
    # z-score the ranks for a standard correlation (avoid catastrophic cancellation)
    ra = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-12)
    rb = (rb - rb.mean()) / (rb.std(unbiased=False) + 1e-12)
    denom = (ra.std(unbiased=False) * rb.std(unbiased=False)).item()
    if denom == 0.0 or torch.isnan(ra).any() or torch.isnan(rb).any():
        return 0.0
    return float((ra * rb).mean().item())

# -------------------------------------------------------------------------
# A-RLVI with Bayes-odds teacher ONLY (no extra KL-to-prior term)
# - CE path uses stop_grad(π) with mean-normalization to break collapse.
# - Teacher q_i = σ( -ℓ_i + log K + logit(π̄) ), where:
#     ℓ_i = per-sample CE (no reduction)
#     K   = #classes = logits.shape[1]
#     π̄   = slow EMA of mean(π_i) in the epoch (scalar, detached)
# - L_φ = mean( KL( Bern(π_i) || Bern(q_i) ) )  (teacher-only)
# -------------------------------------------------------------------------
def train_arlvi_bayes(
    *,
    model_features: torch.nn.Module,
    model_classifier: torch.nn.Module,
    inference_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optim_all: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    ema_alpha: float = 0.95,
    scaler: 'torch.cuda.amp.GradScaler|None' = None,
    return_diag: bool = False,
    log_every: int = 200,
    scheduler: 'torch.optim.lr_scheduler._LRScheduler|None' = None,
    grad_clip: float | None = None,
    pi_bins: Tuple[float, float] = (0.25, 0.75),
):
    """
    Train for ONE epoch with the Bayes-odds teacher (theoretically optimal target),
    but WITHOUT the additional KL(π || π̄) term. This recovers the RLVI posterior
    per-batch (up to the +logK shift included explicitly here) while maintaining
    end-to-end stability via:
      - CE path detached from π (cannot minimize CE by globally shrinking π),
      - EMA π̄ used only inside q_i (teacher), i.e., no gradient path via π̄.

    Returns:
      (CE_avg, KL_avg, acc_avg) or (..., diag) if return_diag=True
    """
    # Make sure all nets are in train mode (BN stats frozen upstream if desired)
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # --- running totals for epoch averages ---
    ce_sum = 0.0         # sum of detached, weighted CE across all samples
    kl_sum = 0.0         # sum of KL(π || q) across all samples
    n_seen = 0
    n_correct = 0

    # grad-norm diagnostics
    grad_bb_sum = grad_cls_sum = grad_inf_sum = 0.0
    grad_snapshots = inf_snapshots = 0

    # π histogram sample (avoid collecting everything to keep memory sane)
    pi_samples, pi_cap = [], 20_000

    # supplemental diagnostics
    spearman_sum = 0.0
    spearman_cnt = 0
    pct_low_sum = pct_high_sum = 0.0
    pct_cnt = 0

    # LR traces for plotting (match zscore implementation)
    lr_trace_bb: list[float] = []
    lr_trace_cls: list[float] = []

    # π→correctness tracking
    bin_lo, bin_hi = pi_bins
    pi_bin_totals = [0, 0, 0]
    pi_bin_correct = [0, 0, 0]

    # Slow EMA prior π̄ (detached scalar). This is *not* a penalty term; it only feeds the teacher.
    pi_bar_running: torch.Tensor | None = None

    # -------------------------------
    # mini-batch loop
    # -------------------------------
    for b_idx, (images, labels, *_) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch:03d}")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # ---- forward ----
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            # 1) features
            z_i = model_features(images)     # [B, D, 1, 1] for ResNet
            z_i = z_i.view(B, -1)            # [B, D]

            # 2) logits
            logits = model_classifier(z_i)    # [B, C]
            C = logits.size(1)                # num classes (K in the derivation)

            # 3) inference: π_i = σ(f_φ(z_i))
            #    IMPORTANT: we do *not* detach z_i here, so φ learns features → π.
            pi_i = inference_net(z_i)         # [B,] or [B,1] depending on your net (keep as [B,])

            # 4) per-sample CE (no reduction); this is ℓ_i in the derivation
            ce_vec = F.cross_entropy(logits, labels, reduction="none")  # [B,]

            # 5) teacher q_i via Bayes-odds (detached):
            #    q_i = σ( -ℓ_i + log K + logit(π̄) )
            #    ℓ_i should not backprop into θ or φ via L_φ, so detach it.
            logK = math.log(C)
            # maintain EMA of mean(π) per batch
            batch_mean_pi = pi_i.mean().detach()
            with torch.no_grad():
                if pi_bar_running is None:
                    pi_bar_running = batch_mean_pi
                else:
                    pi_bar_running = ema_alpha * pi_bar_running + (1.0 - ema_alpha) * batch_mean_pi
            pi_bar = pi_bar_running.detach()  # scalar

            # stable logit(π̄)
            logit_pibar = _logit_clamped(pi_bar)

            # teacher logits: -ℓ_i + logK + logit(π̄)
            teacher_logit = (-ce_vec.detach()) + logK + logit_pibar
            q_i = torch.sigmoid(teacher_logit)  # [B,], DETACHED w.r.t. θ because ce_vec is detached

            # 6) CE path for θ: use stop_grad(π_i) and mean-normalize to keep gradient scale consistent
            pi_w = pi_i.detach() / (batch_mean_pi + 1e-8)   # mean(pi_w) ≈ 1
            L_theta = (pi_w * ce_vec).mean()

            # 7) φ path: teacher-only KL( π || q )
            #    There is NO KL(π || π̄) term here by design.
            kl_to_q = kl_bern(pi_i, q_i)        # [B,]
            L_phi = kl_to_q.mean()

            # accounting for epoch averages (sum over samples)
            ce_term = (pi_w * ce_vec).sum()
            kl_term = kl_to_q.sum()

        # ---- backward / step ----
        optim_all.zero_grad(set_to_none=True)
        if scaler is not None and torch.cuda.is_available():
            scaler.scale(L_theta + L_phi).backward()
            # unscale once for any clipping/diagnostics
            scaler.unscale_(optim_all)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), grad_clip)
        else:
            (L_theta + L_phi).backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), grad_clip)

        # grad norm snapshots before step (for logging)
        if (b_idx + 1) % log_every == 0:
            total_norm_bb  = torch.nn.utils.clip_grad_norm_(model_features.parameters(), float('inf'))
            total_norm_cls = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), float('inf'))
            total_norm_inf = torch.nn.utils.clip_grad_norm_(inference_net.parameters(), float('inf'))
            grad_bb_sum  += float(total_norm_bb)
            grad_cls_sum += float(total_norm_cls)
            grad_inf_sum += float(total_norm_inf)
            grad_snapshots += 1
            inf_snapshots  += 1

            # quick batch log
            tqdm.write(
                f" ↳ bt {b_idx:04d}: "
                f"Lθ={L_theta.item():.3f} Lφ={L_phi.item():.3f} | "
                f"CĒ={ce_vec.mean().item():.3f} KLq̄={kl_to_q.mean().item():.3f} | "
                f"π̄_batch={float(batch_mean_pi):.3f} π̄_ema={float(pi_bar):.3f} | "
                f"π_min={pi_i.min().item():.3f} π_max={pi_i.max().item():.3f} | "
                f"spearman(pi_vs_negCE)={_spearman_corr_torch(pi_i, -ce_vec):.3f} "
                f"pct_pi_below_0.25={(pi_i < 0.25).float().mean().item()*100:.1f}% "
                f"pct_pi_above_0.75={(pi_i > 0.75).float().mean().item()*100:.1f}%"
            )

        # optimizer step + scheduler
        if scaler is not None and torch.cuda.is_available():
            scaler.step(optim_all)
            scaler.update()
        else:
            optim_all.step()

        if scheduler is not None:
            # capture LRs used this batch for plotting (same indices as main)
            try:
                lr_trace_bb.append(optim_all.param_groups[0]['lr'])   # backbone
                lr_trace_cls.append(optim_all.param_groups[2]['lr'])  # head
            except Exception:
                pass
            scheduler.step()

        # ---- running stats ----
        ce_sum += ce_term.item()
        kl_sum += kl_term.item()
        n_seen += B
        n_correct += logits.argmax(1).eq(labels).sum().item()

        # sample π for a final histogram plot
        ps = pi_i.detach().flatten().cpu()
        if ps.numel() and sum(t.numel() for t in pi_samples) < pi_cap:
            pi_samples.append(ps[: min(1024, ps.numel())])

        # supplemental diagnostics
        rho = _spearman_corr_torch(pi_i, -ce_vec)
        spearman_sum += rho
        spearman_cnt += 1
        pct_low_sum  += (pi_i < 0.25).float().mean().item()
        pct_high_sum += (pi_i > 0.75).float().mean().item()
        pct_cnt += 1

        # π→correctness binning
        with torch.no_grad():
            preds = logits.argmax(1)
            correct_mask = preds.eq(labels)
            p = pi_i
            m0 = p < bin_lo
            m1 = (p >= bin_lo) & (p <= bin_hi)
            m2 = p > bin_hi
            pi_bin_totals[0] += int(m0.sum().item())
            pi_bin_totals[1] += int(m1.sum().item())
            pi_bin_totals[2] += int(m2.sum().item())
            if m0.any(): pi_bin_correct[0] += int(correct_mask[m0].sum().item())
            if m1.any(): pi_bin_correct[1] += int(correct_mask[m1].sum().item())
            if m2.any(): pi_bin_correct[2] += int(correct_mask[m2].sum().item())

    # ---- epoch aggregates ----
    ce_epoch = ce_sum / n_seen
    kl_epoch = kl_sum / n_seen
    acc_epoch = n_correct / n_seen

    grad_bb  = grad_bb_sum  / max(1, grad_snapshots)
    grad_cls = grad_cls_sum / max(1, grad_snapshots)
    grad_inf = grad_inf_sum / max(1, inf_snapshots)

    if pi_samples:
        pi_cat = torch.cat(pi_samples)
        pi_diag = {
            "pi_min": float(pi_cat.min()),
            "pi_max": float(pi_cat.max()),
            "pi_mean": float(pi_cat.mean()),
        }
    else:
        pi_diag = {"pi_min": 0.0, "pi_max": 0.0, "pi_mean": 0.0}

    rho_epoch      = spearman_sum / max(1, spearman_cnt)
    pct_low_epoch  = (pct_low_sum  / max(1, pct_cnt)) * 100.0
    pct_high_epoch = (pct_high_sum / max(1, pct_cnt)) * 100.0

    def _safe_div(a: int, b: int) -> float:
        return (a / b) if b > 0 else 0.0

    pi_acc_bins = {
        "lt_0.25":  _safe_div(pi_bin_correct[0], pi_bin_totals[0]),
        "0.25_0.75":_safe_div(pi_bin_correct[1], pi_bin_totals[1]),
        "gt_0.75":  _safe_div(pi_bin_correct[2], pi_bin_totals[2]),
    }
    pi_bin_counts = {
        "lt_0.25":  int(pi_bin_totals[0]),
        "0.25_0.75":int(pi_bin_totals[1]),
        "gt_0.75":  int(pi_bin_totals[2]),
    }

    diag: Dict[str, Any] = {
        "grad_backbone":   grad_bb,
        "grad_classifier": grad_cls,
        "grad_inference":  grad_inf,
        **pi_diag,
        "pi_bar": float(pi_bar_running) if pi_bar_running is not None else 0.0,
        "spearman_rank_pi_vs_negCE": float(rho_epoch),
        "percent_pi_below_0.25": float(pct_low_epoch),
        "percent_pi_above_0.75": float(pct_high_epoch),
        "lr_trace_backbone": lr_trace_bb,
        "lr_trace_classifier": lr_trace_cls,
        "pi_acc_bins": pi_acc_bins,
        "pi_bin_counts": pi_bin_counts,
        "pi_bins": (bin_lo, bin_hi),
    }

    return (ce_epoch, kl_epoch, acc_epoch, diag) if return_diag else (ce_epoch, kl_epoch, acc_epoch)
