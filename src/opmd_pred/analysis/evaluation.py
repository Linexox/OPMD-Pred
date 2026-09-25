import torch

from opmd_pred.analysis.coalitions import coalition_outputs
from opmd_pred.model.ordinal import ordinal_prediction


def classification_metrics(predictions, labels, class_count=3):
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
    qwk = 1.0 if expected_disagreement == 0 else 1.0 - observed_disagreement / expected_disagreement
    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "macro_f1": f1.mean().item(),
        "balanced_accuracy": recall[support > 0].mean().item(),
        "mae": (predictions - labels).abs().float().mean().item(),
        "qwk": qwk.item(),
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
        predictions = ordinal_prediction(logits)
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
