import torch
import torch.nn.functional as F


def compute_kl_divergence(pi_i: torch.Tensor, pi_bar: float) -> torch.Tensor:
    """
    Compute the KL divergence D_KL(Ber(pi_i) || Ber(pi_bar)) for each sample.

    Args:
        pi_i: Tensor of shape (batch_size,), belief scores per sample.
        pi_bar: Scalar, the prior belief of being clean (e.g., 0.5)

    Returns:
        kl: Tensor of shape (batch_size,), KL divergence per sample.
    """
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
    """
    One epoch of training using the A-RLVI method.

    Args:
        model_features: Feature extractor (e.g., ResNet without the final FC layer)
        model_classifier: Final classification head (e.g., Linear layer)
        inference_net: Inference network f_ϕ(zᵢ) predicting πᵢ ∈ (0, 1)
        dataloader: Training data loader
        optimizer: Optimizer for the model parameters
        inference_optimizer: Optimizer for the inference network
        device: torch device (CPU, MPS, CUDA)
        epoch: Current epoch index (for logging/debugging)
        lambda_kl: Weight for the KL regularization term
        pi_bar: Prior belief (default: 0.5)

    Returns:
        total_loss_avg: Average training loss for this epoch
        train_acc: Training accuracy over all batches
    """
    model_features.train()
    model_classifier.train()
    inference_net.train()

    total_correct = 0
    total_seen = 0
    total_loss = 0.0

    for images, labels, _ in dataloader:
        
        images = images.to(device)            # [batch_size, 3, 112, 112]
        labels = labels.to(device)            # [batch_size]
        batch_size = images.size(0)

        # Step 1: Forward pass through feature extractor
        z_i = model_features(images)          # [batch_size, 2048, 1, 1]
        z_i = z_i.view(batch_size, -1)        # Flatten to [batch_size, 2048]

        # Step 2: Get logits from classifier
        logits = model_classifier(z_i)        # [batch_size, num_classes]

        # Step 3: Compute πᵢ from inference network
        pi_i = inference_net(z_i)             # [batch_size], scalar per sample in (0, 1)

        # Step 4: Cross-entropy loss per sample
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [batch_size]

        # Step 5: Compute KL divergence to prior
        kl_loss = compute_kl_divergence(pi_i, pi_bar)                # [batch_size]

        # Step 6: Total loss: weighted CE + KL regularization
        weighted_ce = (pi_i * ce_loss).mean()                        # scalar
        mean_kl = kl_loss.mean()                                     # scalar
        total_loss_batch = weighted_ce + lambda_kl * mean_kl         # scalar

        # Backpropagation
        optimizer.zero_grad()
        inference_optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        inference_optimizer.step()

        # Track stats
        total_loss += total_loss_batch.item() * batch_size
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_seen += batch_size

    avg_loss = total_loss / total_seen
    train_acc = total_correct / total_seen

    return avg_loss, train_acc
