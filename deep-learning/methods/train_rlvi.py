import torch
from torch.nn import functional as F
from utils import accuracy
from tqdm.auto import tqdm



__all__ = ['train_rlvi']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'


@torch.no_grad()
def update_sample_weights(residuals, weights, tol=1e-3, maxiter=40):
    """
    Optimize Bernoulli probabilities
    
    Parameters
    ----------
        residuals : array,
            shape: len(train_loader) - cross-entropy loss for each training sample.
        weights : array,
            shape: len(train_loader) - Bernoulli probabilities pi: pi_i = probability that sample i is non-corrupted.
            Updated inplace.
    """
    residuals.sub_(residuals.min())
    exp_res = torch.exp(-residuals)
    avg_weight = 0.95
    for _ in range(maxiter):
        ratio = avg_weight / (1 - avg_weight)
        new_weights = torch.div(ratio * exp_res, 1 + ratio * exp_res)
        error = torch.norm(new_weights - weights)
        weights[:] = new_weights
        avg_weight = weights.mean()
        if error < tol:
            break

def false_negative_criterion(weights, alpha=0.05):
    '''Find threshold from the fixed probability (alpha) of type II error'''
    total_positive = torch.sum(1 - weights)
    beta = total_positive * alpha
    sorted_weights, _ = torch.sort(weights, dim=0, descending=True)
    false_negative = torch.cumsum(1 - sorted_weights, dim=0)
    last_index = torch.sum(false_negative <= beta) - 1
    threshold = sorted_weights[last_index]
    return threshold


def train_rlvi(train_loader, model, optimizer,
               residuals, weights, overfit, threshold, 
               writer=None, epoch=None):
    """
    Train one epoch: apply SGD updates using Bernoulli probabilities. 
    Thus, optimize variational bound of the marginal likelihood 
    instead of the standard neg. log-likelihood (cross-entropy for classification).
    
    Parameters
    ----------
        residuals : array,
            shape: len(train_loader) - to store cross-entropy loss on each training sample.
            Shared across epochs.
        weights : array,
            shape: len(train_loader) - Bernoulli probabilities: pi_i = proba (in [0; 1]) that sample i is non-corrupted.
            Shared across epochs.
        overfit : bool - flag indicating whether overfitting has started.
        threshold : float in [0; 1] - previous threshold for truncation: pi_i < threshold => pi_i = 0.

    Returns
    -------
        train_acc : float - top-1 accuracy on training samples.
        threshold : float in [0; 1] - updated threshold, based on type II error criterion.
        avg_loss : float - average weighted cross-entropy loss over the epoch.
    """

    train_total = 0
    train_correct = 0
    total_loss = 0.0
    total_seen = 0

    progress = tqdm(train_loader, desc=f"RLVI epoch {epoch}", leave=False)
    for images, labels, indexes in progress:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)

        # Count-based accuracy (robust and easy to read)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # Per-sample CE, then weight by current π (weights)
        loss_vec = F.cross_entropy(logits, labels, reduction='none')
        residuals[indexes] = loss_vec.detach()  # store raw CE for the fixed-point
        batch_weights = weights[indexes]

        loss = (loss_vec * batch_weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)

        # Live progress
        running_acc = 100.0 * train_correct / max(1, train_total)
        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            acc=f"{running_acc:.2f}%",
            pi_mean=f"{weights.mean().item():.3f}"
        )


    update_sample_weights(residuals, weights)
    if overfit:
        # Regularization: truncate samples with high probability of corruption
        threshold = max(threshold, false_negative_criterion(weights))
        weights[weights < threshold] = 0

    train_acc = float(train_correct) / float(train_total)
    avg_loss = total_loss / total_seen

        # Log to TensorBoard if writer is provided
    if writer is not None and epoch is not None:
        writer.add_scalar("Loss/CE", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Inference/MeanPi", weights.mean().item(), epoch)

    return train_acc, threshold, avg_loss
