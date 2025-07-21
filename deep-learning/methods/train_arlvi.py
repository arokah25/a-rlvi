"""
train_arlvi.py
==============

Mini-batch training loop for A-RLVI with practical-stability fixes:

  1. Soft squashing of πᵢ ∈ (0.05,0.95) keeps gradients alive.
  2. rubber-band KL weight starts at 4.0, decays to 1.0 over 5 epochs.
  3. KL prior π̄ is detached from the graph (no gradient into the prior).
  4. π̄ is updated by an EMA **once per epoch** and is detached from the computational graph –-> gives a true regularizing "anchor".
  5. Entropy regularisation is *subtracted* (encourages uncertainty).
    - it is linearly annealed from β=0.4 to 0.0 over epochs 10-20.
    - strong initial entropy term keeps φ from over-reacting and collapsing π to 0 or 1
    - After that we want φ to decide: “this looks corrupt → π→0.1” or “this is clean → π→0.9”
    - Reducing β removes the tug toward 0.5, letting CE and KL separate the two groups—hence a more bimodal posterior.
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
    pi_bar:        float = 0.9,   # warm-up prior value 90% clean, 10% corrupt
    warmup_epochs: int   = 2,
    alpha:         float = 0.97,  # EMA momentum for π̄ after warm-up
    pi_bar_ema:    float = 0.9,   # running prior coming in from previous epoch
    beta:          float = 0.4,   # initial weight on entropy regularisation before decay
    tau:           float = 0.6,   # CE-temperature (0<tau<1)
    scheduler=None,  # optional learning rate scheduler
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
        # During warm-up we **freeze φ** by disabling gradient tracking
        if epoch < warmup_epochs:
            with torch.no_grad():
                pi_raw = inference_net(z_i)    # sigmoid inside net → (0,1)
        else:
            pi_raw = inference_net(z_i)

        pi_i = 0.8 * pi_raw + 0.1     # (0.1, 0.9) keeps gradients alive

        all_pi_values.append(pi_i.detach().cpu())

        # ------------- per-sample CE --------------------------------
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)

        # ------------- KL divergence --------------------------------
        pi_bar_tensor = torch.full_like(pi_i, prior_value).detach()  # no grad
        kl_loss = compute_kl_divergence(pi_i, pi_bar_tensor)         # (B,)

        # ------------- Entropy regularisation -----------------------
        #First we linearly anneal the beta value from 0.4 to 0.0 over the epochs
        # This is done to encourage exploration at the start of training.
        decay_start = 12
        decay_length = 10

        if epoch >= decay_start:
            frac       = max(0.0, 1.0 - (epoch - decay_start) / decay_length)
            beta_now   = beta * frac
        else:
            beta_now   = beta

        entropy = -(pi_i*torch.log(pi_i+eps) + (1-pi_i)*torch.log(1-pi_i+eps))  
        entropy_reg = beta_now * entropy.mean()          # subtract later

        # ------------- Loss composition -----------------------------
        #ce_weighted = (pi_i * ce_loss).sum() / B
        #mean_kl     = kl_loss.mean()
        pi_temp     = pi_i ** tau         # temperature scaling  (τ<1) with concave power
        ce_weighted = (pi_temp * ce_loss).sum() / B
        mean_kl     = kl_loss.mean()


        # ----- KL rubber-band: ×3 right after warm-up, then decay ------------------
        # λₖₗ is the base KL weight, e.g. 2.0
        # λₖₗ is multiplied by a schedule value that starts at 4.
        # The schedule value decays linearly from 4.0 to 1.0
        # over the next 5 epochs, but never drops below λₖₗ.
        # This gives a strong KL penalty at the start, then decays it
        # to a stable value that is still above the base KL weight.
        # -----------------------------------------------------------------
        base_kl = lambda_kl 

        # rubber-band schedule value: 4.0, 3.4, 2.8, 2.2, 1.6, 1.0 …
        epochs_after = max(epoch - warmup_epochs, 0)
        schedule_val = 4.0 - 0.6 * epochs_after

        # effective KL weight: starts at 4, never drops below base_kl
        kl_weight = max(base_kl, schedule_val)

        total_batch_loss = (
            ce_weighted                           # additive CE
            + kl_weight * mean_kl                 # KL with rubber-band
            - entropy_reg                         # entropy bonus
        )

        # ------------- Back-prop ------------------------------------
        optimizer.zero_grad()
        if epoch >= warmup_epochs:
            inference_optimizer.zero_grad()

        total_batch_loss.backward()

        if epoch >= warmup_epochs:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(inference_net.parameters(), grad_clip)
            optimizer.step()             # θ update
            inference_optimizer.step()   # φ update
        else:
            optimizer.step()             # θ update only (φ frozen)

        # --- batch‐wise LR update for OneCycleLR ---
        if scheduler is not None:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LR/main", current_lr, epoch * steps_per_epoch + batch_idx)



        # ------------- Stats ----------------------------------------
        total_loss += total_batch_loss.item() * B
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
            # inference‐net
            writer.add_scalar("GradNorm/Inference", grad_inf, epoch * len(dataloader) + batch_idx)

            # classifier (θ)
            clf_norm = torch.nn.utils.clip_grad_norm_(model_classifier.parameters(), float('inf')).item()
            writer.add_scalar("GradNorm/Classifier",  clf_norm, epoch * len(dataloader) + batch_idx)


            print(f"[Epoch {epoch:02d} Batch {batch_idx:04d}] "
                  f"πᵢ μ={pi_i.mean():.3f} min={pi_i.min():.2f} max={pi_i.max():.2f} "
                  f"CE={ce_weighted.item():.3f}  KL={mean_kl.item():.3f}  "
                  f"|∇φ|={grad_inf:.2f}"
                  f"|∇θ|={clf_norm:.2f}"
                  f" ce_loss={total_ce / total_seen:.3f} "
                  f" kl_loss={total_kl / total_seen:.3f} "
                  f" train_acc={total_correct / total_seen:.3f} "
                  f" lr={current_lr:.6f} "
                  f" pi_bar_ema={pi_bar_ema:.3f} "
                  f" β={beta_now:.3f} ")

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
