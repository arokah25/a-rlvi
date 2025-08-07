# train_arlvi_vanilla.py
# ==============================================================
# A *minimal* A-RLVI training loop meant for debugging theory.
# --------------------------------------------------------------
# ‣   A single hyper-parameter  `beta`  that scales the KL term
#     (analogous to β-VAE) so we can study CE⇆KL balance cleanly.
# ‣   Optional update schedule for the inference net:
#       • 'batch'  – update ϕ every mini-batch (default).
#       • 'epoch'  – freeze ϕ for an epoch, update once at the end.
# ‣   Logs the *raw magnitudes* of CE and KL **and** their gradient
#     norms so we can see which term dominates and when.
# --------------------------------------------------------------
# Usage example (pseudo-CLI):
#   python main.py --trainer vanilla \
#                  --beta 1.0         \
#                  --update_inference_every epoch
# ==============================================================

from __future__ import annotations
import torch, torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility: KL( Bern(π) ‖ Bern(π̄) ) element-wise for matching shapes
# -----------------------------------------------------------------------------
def kl_bern(pi: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """Numerically-stable KL divergence between two Bernoulli parameters."""
    eps = 1e-6
    pi     = pi.clamp(eps, 1. - eps)
    pi_bar = pi_bar.clamp(eps, 1. - eps)
    return pi * (pi.log() - pi_bar.log()) + (1. - pi) * ((1. - pi).log() - (1. - pi_bar).log())

# -----------------------------------------------------------------------------
# Main training routine (single epoch)
# -----------------------------------------------------------------------------
def train_arlvi_vanilla(
    *,
    model_features:   torch.nn.Module,          # backbone   (θ_bb)
    model_classifier: torch.nn.Module,          # classifier (θ_cls)
    inference_net:    torch.nn.Module,          # f_ϕ
    dataloader:       torch.utils.data.DataLoader,
    optim_backbone:   torch.optim.Optimizer,
    optim_classifier: torch.optim.Optimizer,
    optim_inference:  torch.optim.Optimizer,
    device:           torch.device,
    epoch:            int,
    beta:             float = 0.75,             # <-- single hyper-parameter!
    update_inference_every: str = "batch",      # 'batch' | 'epoch'
    clamp_min:        float = 0.05,             # keep π in (clamp_min, 1-clamp_min)
    return_diag:      bool  = False,            # return diagnostic dict if True
    log_every:        int   = 200,              # mini-batch console logging
) -> Tuple[float, float, float] | Tuple[float, float, float, Dict[str, float]]:
    """Train A-RLVI for one epoch.

    Returns
    -------
    (CE_avg, KL_avg, acc_avg)            – default
    (CE_avg, KL_avg, acc_avg, diag)      – if return_diag=True
    """

    # ------------------------------------------------------
    # Put nets in train mode
    # ------------------------------------------------------
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # ------------------------------------------------------
    # Running sums
    # ------------------------------------------------------
    ce_sum = kl_sum = 0.0
    n_seen = n_correct = 0

    grad_bb_sum = grad_cls_sum = 0.0          # accumulate batch means
    grad_inf_sum, n_inf_batches = 0.0, 0      # keep separate counter

    # π sampling buffer (memory-safe)
    pi_samples, pi_cap = [], 20_000

    # Zero grads in once-per-epoch mode
    if update_inference_every == "epoch":
        optim_inference.zero_grad(set_to_none=True)

    # ------------------------------------------------------
    # Mini-batch loop
    # ------------------------------------------------------
    for b_idx, (images, labels, *_) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch:03d}")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B      = images.size(0)

        # ---------- forward ------------------------------------------
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            z_i    = model_features(images).view(B, -1) # backbone features shape (B, D)
            logits = model_classifier(z_i) # shape (B, C)
            pi_i   = inference_net(z_i).clamp(clamp_min, 1. - clamp_min) # shape (B,)
            #pi_bar = pi_i.mean().detach()

            ce_vec = F.cross_entropy(logits, labels, reduction="none")
            kl_vec = kl_bern(pi_i, torch.full_like(pi_i, 0.98))

            loss = ((pi_i * ce_vec).mean() + beta * kl_vec.mean())


        # ---------- backward -----------------------------------------
        for opt in (optim_backbone, optim_classifier):
            opt.zero_grad(set_to_none=True)
        if update_inference_every == "batch":
            optim_inference.zero_grad(set_to_none=True)

        loss.backward()

        optim_backbone.step()
        optim_classifier.step()
        if update_inference_every == "batch":
            optim_inference.step()

        # ---------- stats --------------------------------------------
        ce_sum += (pi_i * ce_vec).sum().item()
        kl_sum += kl_vec.sum().item()
        n_seen += B
        n_correct += logits.argmax(1).eq(labels).sum().item()

        # grad norms (per-batch means, not scaled by B)
        grad_bb_batch  = sum(p.grad.norm().item() for p in model_features.parameters()   if p.grad is not None)
        grad_cls_batch = sum(p.grad.norm().item() for p in model_classifier.parameters() if p.grad is not None)
        grad_bb_sum  += grad_bb_batch
        grad_cls_sum += grad_cls_batch

        if update_inference_every == "batch":
            grad_inf_batch = sum(p.grad.norm().item() for p in inference_net.parameters() if p.grad is not None)
            grad_inf_sum   += grad_inf_batch
            n_inf_batches  += 1

        # lightweight π sampling
        ps = pi_i.detach().flatten().cpu()
        if ps.numel() and sum(t.numel() for t in pi_samples) < pi_cap:
            pi_samples.append(ps[: min(1024, ps.numel())])

        # ---------- per-batch console log -------------------
        if (b_idx + 1) % log_every == 0:
            tqdm.write(f"  ↳ bt {b_idx:04d}: loss={loss.item():.3f} "
                       f"CE={ce_vec.mean():.3f} KL={kl_vec.mean():.3f}"
                       f"  | grad_bb={grad_bb_batch:.3f} "
                       f"grad_cls={grad_cls_batch:.3f} "
                       f"grad_inf={grad_inf_batch:.3f} "
                       f"π_min={pi_i.min():.3f} "
                          f"π_max={pi_i.max():.3f} ")

    # ------------------------------------------------------
    # One-shot inference-net update  (epoch mode)
    # ------------------------------------------------------
    if update_inference_every == "epoch":
        # Manually scale the accumulated gradients of ϕ by 1 / len(dataloader)
        for p in inference_net.parameters():
            if p.grad is not None:
                p.grad.div_(len(dataloader))  # average over all batches

        optim_inference.step()

        # Diagnostics: compute grad norm for inference net
        grad_inf_sum  = sum(p.grad.norm().item() for p in inference_net.parameters() if p.grad is not None)
        n_inf_batches = 1


    # ------------------------------------------------------
    # Averages
    # ------------------------------------------------------
    ce_epoch  = ce_sum / n_seen
    kl_epoch  = kl_sum / n_seen
    acc_epoch = n_correct / n_seen

    grad_bb  = grad_bb_sum  / len(dataloader)
    grad_cls = grad_cls_sum / len(dataloader)
    grad_inf = grad_inf_sum / max(1, n_inf_batches)


    # ------------------------------------------------------
    # Diagnostics dict
    # ------------------------------------------------------
    if pi_samples:
        pi_cat = torch.cat(pi_samples)
        pi_diag = {
            "pi_min":  float(pi_cat.min()),
            "pi_max":  float(pi_cat.max()),
            "pi_mean": float(pi_cat.mean()),
        }
    else:
        pi_diag = {"pi_min": 0.0, "pi_max": 0.0, "pi_mean": 0.0}

    diag = {
        "grad_backbone":   grad_bb,
        "grad_classifier": grad_cls,
        "grad_inference":  grad_inf,
        **pi_diag,
    }

    if return_diag:
        return ce_epoch, kl_epoch, acc_epoch, diag
    else:
        return ce_epoch, kl_epoch, acc_epoch

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
