import torch
import torch.nn.functional as F


def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: float) -> torch.Tensor:
    eps = 1e-6  # for numerical stability
    pi_i = pi_i.clamp(eps, 1 - eps)
    pi_bar = torch.tensor(pi_bar, device=pi_i.device)
    kl = pi_i * torch.log(pi_i / pi_bar) + (1 - pi_i) * torch.log((1 - pi_i) / (1 - pi_bar))
    return kl


def train_arlvi(
    model_features: torch.nn.Module,
    model_classifier: torch.nn.Module,
    inference_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    inference_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lambda_kl: float = 1.0,
    pi_bar: float = 0.5,
):
    model_features.train()
    model_classifier.train()
    inference_net.train()

    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0

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

        # Step 4: Per-sample cross-entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]

        # Step 5: KL divergence
        kl_loss = compute_kl_divergence(pi_i, pi_bar)                # [B]

        # Step 6: Total loss
        weighted_ce = (pi_i * ce_loss).mean()                        # scalar
        mean_kl = kl_loss.mean()                                     # scalar
        total_loss_batch = weighted_ce + lambda_kl * mean_kl         # scalar

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
        if batch_idx % 100 == 0:
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
    return avg_loss, avg_ce_loss, avg_kl_loss, train_acc
