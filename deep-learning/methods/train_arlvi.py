import torch
import torch.nn.functional as F


def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: float) -> torch.Tensor:
    eps = 1e-6  # for numerical stability
    pi_i = pi_i.clamp(eps, 1 - eps)
    pi_bar = torch.full_like(pi_i, fill_value=pi_bar)
    kl = pi_i * torch.log(pi_i / pi_bar) + (1 - pi_i) * torch.log((1 - pi_i) / (1 - pi_bar))
    return kl


def train_arlvi(
    model_features: torch.nn.Module,  # feature extractor, e.g., ResNet50
    model_classifier: torch.nn.Module,  # classifier, e.g., MLP on top of features
    inference_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    inference_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lambda_kl: float = 1.0,
    pi_bar: float = 0.5,
    warmup_epochs: int = 2,
    writer=None
    ):

    model_features.train()
    model_classifier.train()
    inference_net.train()

    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    all_pi_values = []  # collect πᵢ across batches for histogram

    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Step 1: Forward pass through feature extractor
        z_i = model_features(images)          # [B, 2048, 1, 1]
        z_i = z_i.view(batch_size, -1)        # [B, 2048]

        # Step 2: Get logits
        logits = model_classifier(z_i)        # [B, num_classes]

        # Step 3: Compute πᵢ
        pi_i = inference_net(z_i).clamp(1e-6, 1 - 1e-6)  # [B]
        all_pi_values.append(pi_i.detach().cpu())        # accumulate for histogram

        # Step 4: Per-sample cross-entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]

        # Step 5: KL divergence
        if epoch < warmup_epochs:  # Warm-up phase (for training stability)
            kl_loss = compute_kl_divergence(pi_i, pi_bar)  # scalar prior (default = 0.9)
        else:  # Use empirical prior from πᵢ in the current batch (true variational inference)
            pi_bar_empirical = pi_i.detach().mean()
            kl_loss = compute_kl_divergence(pi_i, pi_bar_empirical)  # [B]

        # Step 6: Total loss
        weighted_ce = (pi_i * ce_loss).mean()              # scalar
        mean_kl = kl_loss.mean()                           # scalar
        total_loss_batch = weighted_ce + lambda_kl * mean_kl  # scalar

        # Backward + optimize
        optimizer.zero_grad()
        inference_optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        inference_optimizer.step()

        # Stats
        total_loss += total_loss_batch.item() * batch_size
        total_ce += weighted_ce.item() * batch_size
        total_kl += mean_kl.item() * batch_size
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_seen += batch_size

        # -------- DEBUG LOGGING --------
        if batch_idx % 500 == 0:
            grad_norm = 0.0
            for param in inference_net.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            print(f"[Epoch {epoch} | Batch {batch_idx}]")
            print(f"  πᵢ stats: mean={pi_i.mean().item():.4f}, min={pi_i.min().item():.4f}, max={pi_i.max().item():.4f}")
            print(f"  Weighted CE: {weighted_ce.item():.4f}, Mean KL: {mean_kl.item():.4f}")
            print(f"  Inference grad norm: {grad_norm:.4f}")
            print(f"  Total loss: {total_loss_batch.item():.4f}, CE loss: {weighted_ce.item():.4f}, KL loss: {mean_kl.item():.4f}")
        # ------------------------------

    avg_loss = total_loss / total_seen
    avg_ce_loss = total_ce / total_seen
    avg_kl_loss = total_kl / total_seen
    train_acc = total_correct / total_seen
    mean_pi_i = torch.cat(all_pi_values, dim=0).mean().item()

    if writer is not None:
        writer.add_scalar("Loss/CE", avg_ce_loss, epoch)
        writer.add_scalar("Loss/KL", avg_kl_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Inference/MeanPi", mean_pi_i, epoch)

        # Log πᵢ histogram every 10 epochs
        if epoch % 10 == 0:
            pi_concat = torch.cat(all_pi_values, dim=0)  # [N]
            writer.add_histogram("Inference/PiDistribution", pi_concat, epoch)

    return avg_ce_loss, avg_kl_loss, train_acc, mean_pi_i
