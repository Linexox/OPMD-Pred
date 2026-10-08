"""Image-only integrated gradients; tabular observations and availability stay fixed."""

import torch
from PIL import Image, ImageOps

from opmd_pred.model.ordinal import model_probabilities


TARGETS = ("severe", "at_least_moderate", "expected_severity")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def target_value(model, batch, target):
    probabilities = model_probabilities(model(batch), model.head_kind)
    if target == "severe":
        return probabilities[:, 2]
    if target == "at_least_moderate":
        return probabilities[:, 1:].sum(dim=1)
    if target == "expected_severity":
        return probabilities[:, 1] + 2 * probabilities[:, 2]
    raise ValueError(f"Choose a target from {TARGETS}")


def integrated_gradients(model, batch, target="severe", baseline="mean", steps=32):
    """Trapezoidal path integral using steps intervals and steps+1 gradients."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if target not in TARGETS or baseline not in {"mean", "black"}:
        raise ValueError("Unknown attribution target or baseline")
    if model.training:
        raise ValueError("Restore the fixed model in eval mode before attribution")
    image = batch["image"].detach()
    if image.ndim != 4 or image.shape[:2] != (1, 3) or not image.is_floating_point():
        raise ValueError("Attribution requires one floating RGB image [1,3,H,W]")
    if not torch.isfinite(image).all():
        raise ValueError("Image contains non-finite values")
    if "image" not in model.fusion.names or batch["image_present"].numel() != 1 or not batch["image_present"].bool().item():
        raise ValueError("资料不足：没有真实已观测且选入模型的图像")
    reference = torch.zeros_like(image)
    if baseline == "black":
        mean = image.new_tensor(MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(STD).view(1, 3, 1, 1)
        reference = (-mean / std).expand_as(image)
    difference = image - reference
    total_gradient = torch.zeros_like(image)
    endpoints = {}
    # Calling the full model retains the encoder-to-input gradient, even if frozen.
    # autograd.grad requests input gradients only; no parameter .grad accumulation.
    with torch.enable_grad():
        for index in range(steps + 1):
            interpolated = (reference + (index / steps) * difference).detach().requires_grad_(True)
            value = target_value(model, {**batch, "image": interpolated}, target)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite attribution target along path")
            gradient, = torch.autograd.grad(value.sum(), interpolated)
            if not torch.isfinite(gradient).all():
                raise ValueError("Non-finite image gradient along path")
            total_gradient += gradient.detach() * (0.5 if index in {0, steps} else 1.0)
            if index in {0, steps}:
                endpoints[index] = value.detach().item()
    attribution = (difference * total_gradient / steps).detach().cpu()[0]
    attribution_sum = attribution.double().sum().item()
    delta = endpoints[steps] - endpoints[0]
    return {"pixel_attribution": attribution, "target": target, "baseline": baseline,
            "steps": steps, "integration": "trapezoidal; steps intervals; steps+1 gradients",
            "input_value": endpoints[steps], "baseline_value": endpoints[0],
            "output_difference": delta, "attribution_sum": attribution_sum,
            "completeness_residual": delta - attribution_sum,
            "units": "severity grade" if target == "expected_severity" else "probability",
            "image_present_along_entire_path": True}


def image_geometry(original, encoder):
    """Use Pillow's own contain sizing and pad's round-centered offsets."""
    if encoder == "dinov2_b14":
        size = (518, 518)
        fitted = ImageOps.contain(original, size, method=Image.Resampling.BILINEAR)
        left = round((size[0] - fitted.width) * 0.5)
        top = round((size[1] - fitted.height) * 0.5)
        box = (left, top, left + fitted.width, top + fitted.height)
        mode = "PIL.ImageOps.pad/BILINEAR/center=(0.5,0.5)/rgb=(124,116,104)"
    elif encoder == "vit_b16":
        size, box, mode = (224, 224), (0, 0, 224, 224), "torchvision.Resize((224,224)); stretch"
    else:
        raise ValueError("Unknown checkpoint image encoder")
    return {"original_size_wh": list(original.size), "canvas_size_wh": list(size),
            "content_box_ltrb": list(box), "transform": mode}


def save_heatmap(path, original, pixel_attribution, geometry, units):
    """Visualization interpolation is not a mass-preserving attribution transform."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    signed = pixel_attribution.double().sum(dim=0)
    width, height = geometry["canvas_size_wh"]
    if tuple(signed.shape) != (height, width):
        raise ValueError("Attribution shape does not match checkpoint preprocessing")
    left, top, right, bottom = geometry["content_box_ltrb"]
    content = signed[top:bottom, left:right]
    mapped = Image.fromarray(content.float().numpy()).resize(original.size, Image.Resampling.BILINEAR)
    maximum = signed.abs().max().item()
    scale = maximum if maximum > 0 else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title("Original RGB")
    axes[1].imshow(original)
    axes[1].imshow(mapped, cmap="coolwarm", vmin=-scale, vmax=scale, alpha=0.55)
    axes[1].set_title("Signed IG overlay (display interpolation)")
    displayed = axes[2].imshow(signed.numpy(), cmap="coolwarm", vmin=-scale, vmax=scale)
    axes[2].add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top,
                               fill=False, edgecolor="black", linewidth=1.5))
    axes[2].set_title("Model canvas: outside box = padding")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(displayed, ax=axes, label=f"Signed channel-summed IG ({units})")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"display_scale_absolute_max": maximum, "display_colormap": "coolwarm; signed, symmetric",
            "display_mapping": "crop content box; PIL bilinear resize; not sum preserving",
            "content_attribution_sum": content.sum().item(),
            "padding_attribution_sum": signed.sum().item() - content.sum().item(),
            "padding_absolute_attribution_sum": (
                pixel_attribution.double().abs().sum().item()
                - pixel_attribution[:, top:bottom, left:right].double().abs().sum().item())}
