import torch
import torch.nn.functional as F
from amortized.inference_net import InferenceNet
from utils import accuracy
from tqdm import tqdm

def train_arlvi(train_loader, model, optimizer, inference_net, optimizer_inf, args):
    """
    Trains one epoch using the A-RLVI framework.
    This version uses an amortized inference network f_ϕ to learn corruption probabilities πᵢ = σ(f_ϕ(zᵢ)),
    where zᵢ is the input data (e.g., flattened image). The loss consists of a weighted likelihood term
    and a KL divergence regularizer encouraging πᵢ to remain close to the batch average ⟨π⟩.

    Parameters
    ----------
    train_loader : DataLoader
        A PyTorch DataLoader yielding (image, label, index) tuples.
    model : torch.nn.Module
        The classifier model (θ) to train.
    optimizer : torch.optim.Optimizer
        Optimizer for model θ.
    args : argparse.Namespace
        Parsed command-line arguments, including lr_init, batch size, etc.

    Returns
    -------
    train_acc : float
        Top-1 training accuracy over the entire epoch.
    """

    # Get device (e.g., "cuda" or "cpu") from the model
    device = next(model.parameters()).device
    model.train()

    # Determine input dimension by flattening one image tensor of shape (C, H, W)
    # For MNIST: (1, 28, 28) => 784 elements
    input_dim = train_loader.dataset[0][0].numel()
    
    # Track total number of correct predictions and total samples seen
    train_total = 0
    train_correct = 0

    # Iterate over training batches
    for (images, labels, _) in tqdm(train_loader, desc="Training A-RLVI", leave=False):
        # images: shape (B, C, H, W), labels: shape (B,)
        # Move to correct device
        images = images.to(device)
        labels = labels.to(device)

        # Flatten images to shape (B, input_dim) for input to inference network
        z = images.view(images.size(0), -1)  # zᵢ ∈ ℝ^input_dim for each i

        # Zero gradients for both model θ and inference net ϕ
        optimizer.zero_grad()
        optimizer_inf.zero_grad()

        # Forward pass through classification model
        # logits: shape (B, num_classes)
        logits = model(images)

        # Compute per-sample cross-entropy: shape (B,)
        # ℓ_θ(zᵢ) = -log p(yᵢ | zᵢ; θ)
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')

        # Forward pass through inference net: πᵢ = σ(f_ϕ(zᵢ))
        # pi: shape (B,)
        pi = inference_net(z).squeeze()

        # Clamp πᵢ to avoid log(0) in KL divergence computation
        pi_clamped = pi.clamp(min=1e-4, max=1 - 1e-4)  # shape (B,)

        # Compute batch average of πᵢ: scalar
        pi_mean = pi_clamped.mean().detach()

        # Weighted negative log-likelihood: πᵢ ⋅ ℓ_θ(zᵢ), shape (B,)
        weighted_loss = pi_clamped * loss_per_sample

        # KL divergence between Bern(πᵢ) and Bern(⟨π⟩), shape (B,)
        kl_term = pi_clamped * torch.log(pi_clamped / pi_mean) + \
                  (1 - pi_clamped) * torch.log((1 - pi_clamped) / (1 - pi_mean))

        # Total loss: scalar
        loss = weighted_loss.mean() + kl_term.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer_inf.step()

        # Prediction and accuracy tracking
        # preds: shape (B,), top-1 predictions
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Compute overall training accuracy
    train_acc = float(train_correct) / float(train_total)
    return train_acc
