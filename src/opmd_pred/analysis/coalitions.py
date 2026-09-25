import torch


def coalition_masks(modality_count):
    return list(range(1 << modality_count))


def apply_coalition(present, coalition, names):
    masked = {}
    for index, name in enumerate(names):
        masked[name] = present[name] & bool(coalition & (1 << index))
    return masked


@torch.no_grad()
def coalition_outputs(model, batch, device):
    embeddings, observed = model.fusion.modality_embeddings(batch)
    names = model.fusion.names
    outputs = []
    for coalition in coalition_masks(len(names)):
        present = apply_coalition(observed, coalition, names)
        logits = model.forward_from_embeddings(embeddings, present)
        outputs.append(logits)
    return torch.stack(outputs, dim=1), names
