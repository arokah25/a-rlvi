# train_arlvi_vanilla.py
# ==============================================================
# A *minimal* A‑RLVI training loop meant for debugging theory.
# --------------------------------------------------------------
# ‣   A single hyper‑parameter  `beta`  that scales the KL term
#     (analogous to β‑VAE) so we can study CE⇆KL balance cleanly.
# ‣   Optional update schedule for the inference net:
#       • 'batch'  – update ϕ every mini‑batch (default).
#       • 'epoch'  – freeze ϕ for an epoch, update once at the end.
# ‣   Logs the *raw magnitudes* of CE and KL **and** their gradient
#     norms so we can see which term dominates and when.
# --------------------------------------------------------------
# Usage example (pseudo‑CLI):
#   python main.py --trainer vanilla \
#                  --beta 1.0         \
#                  --update_inference_every epoch
# ==============================================================

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility: KL( Bern(π) ‖ Bern(π̄) ) element‑wise for matching shapes
# -----------------------------------------------------------------------------

def kl_bern(pi: torch.Tensor, pi_bar: torch.Tensor) -> torch.Tensor:
    """Numerically‑stable KL divergence between two Bernoulli parameters."""
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
    beta:             float = 1.0,              # <‑‑ single hyper‑parameter!
    update_inference_every: str = "batch",     # 'batch' | 'epoch'
    clamp_min:        float = 0.05,             # keep π in (clamp_min, 1‑clamp_min)
    writer=None                                    # optional TensorBoard
) -> Tuple[float, float, float]:
    """Train A‑RLVI for one epoch (minimal version).

    Returns
    -------
    ce_epoch, kl_epoch, accuracy_epoch
    """

    # ------------------------------------------------------
    # Set all nets to train mode
    # ------------------------------------------------------
    model_features.train()
    model_classifier.train()
    inference_net.train()

    # ------------------------------------------------------
    # Running totals for epoch‑level stats
    # ------------------------------------------------------
    ce_sum = kl_sum = total_loss_sum = 0.0
    n_seen = n_correct = 0

    # For gradient‑norm logging
    grad_norm_bb_sum = grad_norm_cls_sum = grad_norm_inf_sum = 0.0

    # Make sure inference optimiser grads are zero if we update once/epoch
    if update_inference_every == "epoch":
        optim_inference.zero_grad(set_to_none=True)

    # Loop through the mini-batches
    for images, labels, *_ in tqdm(dataloader, desc=f"Epoch {epoch:03d}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # ---------------------------- forward -----------------------------
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            z_i     = model_features(images).view(B, -1)      # backbone features
            logits  = model_classifier(z_i)                   # class logits
            pi_i    = inference_net(z_i).clamp(clamp_min, 1. - clamp_min)  # corruption probs in (δ,1‑δ)

            ce_vec  = F.cross_entropy(logits, labels, reduction="none")     # per‑sample CE
            kl_vec  = kl_bern(pi_i, torch.full_like(pi_i, 0.75))            # fixed scalar prior 0.75

            # --- β‑scaled loss ------------------------------------------
            loss = (pi_i * ce_vec).mean() + beta * kl_vec.mean()

        # --------------------------- backward ---------------------------
        for opt in (optim_backbone, optim_classifier):
            opt.zero_grad(set_to_none=True)
        if update_inference_every == "batch":
            optim_inference.zero_grad(set_to_none=True)

        loss.backward()

        # ------ optimisation steps -------------------------------------
        optim_backbone.step()
        optim_classifier.step()
        if update_inference_every == "batch":
            optim_inference.step()

        # -------------------- running stats ----------------------------
        ce_sum           += (pi_i * ce_vec).sum().item()
        kl_sum           += kl_vec.sum().item()
        total_loss_sum   += loss.item() * B
        n_seen           += B
        n_correct        += logits.argmax(dim=1).eq(labels).sum().item()

        # --- gradient norms (after step so grads are fresh) ------------
        grad_norm_bb_sum   += torch.nn.utils.clip_grad_norm_(model_features.parameters(),   1e9).item() * B
        grad_norm_cls_sum  += torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), 1e9).item() * B
        if update_inference_every == "batch":
            grad_norm_inf_sum += torch.nn.utils.clip_grad_norm_(inference_net.parameters(),  1e9).item() * B

    # ------------------------------------------------------------------
    # One‑shot update of inference net if we froze it for entire epoch
    # ------------------------------------------------------------------
    if update_inference_every == "epoch":
        optim_inference.step()
        grad_norm_inf_sum = torch.nn.utils.clip_grad_norm_(inference_net.parameters(), 1e9).item() * n_seen

    # ---------------------- epoch averages ----------------------------
    ce_epoch   = ce_sum / n_seen
    kl_epoch   = kl_sum / n_seen
    acc_epoch  = n_correct / n_seen

    grad_bb = grad_norm_bb_sum  / n_seen
    grad_cls= grad_norm_cls_sum / n_seen
    grad_inf= grad_norm_inf_sum / n_seen

    # Console log in one line (easy grep)
    print(f"[ep {epoch:03d}] CE {ce_epoch:.4f} | KL {kl_epoch:.4f} | β {beta:.2f} | "
          f"|∇| bbk {grad_bb:.2f} cls {grad_cls:.2f} inf {grad_inf:.2f} | acc {acc_epoch:.2%}")

    # Optional TensorBoard logging
    if writer is not None:
        writer.add_scalars("Loss", {"CE_weighted": ce_epoch, "KL": kl_epoch}, epoch)
        writer.add_scalars("GradNorm", {
            "backbone": grad_bb,
            "classifier": grad_cls,
            "inference": grad_inf,
        }, epoch)
        writer.add_scalar("Accuracy/train", acc_epoch, epoch)
        writer.flush()

    return ce_epoch, kl_epoch, acc_epoch

# -----------------------------------------------------------------------------
# End of file – keep this module small so theory experiments are painless
# -----------------------------------------------------------------------------
