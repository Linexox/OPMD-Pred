import torch

from opmd_pred.analysis.coalitions import coalition_outputs
from opmd_pred.model.ordinal import model_probabilities


def classification_metrics(predictions, labels, class_count=3):
    if labels.numel() == 0:
        raise ValueError("Cannot evaluate an empty split")
    confusion = torch.bincount(
        labels * class_count + predictions,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count).float()
    support = confusion.sum(dim=1)
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    total = confusion.sum().clamp_min(1)
    observed = torch.arange(class_count, device=confusion.device, dtype=torch.float32)
    weights = (observed[:, None] - observed[None, :]).square()
    observed_disagreement = (weights * confusion / total).sum()
    expected = support[:, None] * confusion.sum(dim=0)[None, :] / total
    expected_disagreement = (weights * expected / total).sum()
    qwk = None if expected_disagreement.item() == 0 else (1.0 - observed_disagreement / expected_disagreement).item()
    distance = predictions - labels
    severe_count = int((labels == 2).sum().item())
    return {
        "evaluation_unit": "record",
        "accuracy": (predictions == labels).float().mean().item(),
        "macro_f1": f1.mean().item(),
        "balanced_accuracy": recall[support > 0].mean().item(),
        "mae": (predictions - labels).abs().float().mean().item(),
        "qwk": qwk,
        "count": int(labels.numel()),
        "support": support.to(torch.long).tolist(),
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "underestimate_rate": (distance < 0).float().mean().item(),
        "overestimate_rate": (distance > 0).float().mean().item(),
        "two_level_error_rate": (distance.abs() == 2).float().mean().item(),
        "severe_to_mild_count": int(((labels == 2) & (predictions == 0)).sum().item()),
        "severe_to_mild_rate": int(((labels == 2) & (predictions == 0)).sum().item()) / severe_count if severe_count else None,
        "severe_recall": recall[2].item() if severe_count else None,
        "confusion_matrix": confusion.to(torch.long).tolist(),
    }


@torch.no_grad()
def evaluate_coalitions(model, loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_outputs = []
    names = None
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        logits, names = coalition_outputs(model, batch, device)
        predictions = model_probabilities(logits, model.head_kind).argmax(dim=-1)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["label"].cpu())
        all_outputs.append(logits.cpu())
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_outputs)
    metrics = {
        str(coalition): classification_metrics(predictions[:, coalition], labels)
        for coalition in range(logits.size(1))
    }
    return logits, labels, names, metrics
