# DINOv2 last4 encoder transfer artifact

Artifact: dinov2_last4_epoch3_encoder.pt

This is the headless encoder state from the fixed best-epoch-3 DINOv2-B/14 last-four-block checkpoint. It contains 174 encoder tensors and metadata only; it contains no classifier head, full-model state, patient fields, category labels, or split manifest.

- Model: timm/vit_base_patch14_dinov2.lvd142m
- Public snapshot revision: 4685c99dabffe5affac90bd99dbffd25801ae58d
- Public safetensors SHA-256: 55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b
- Source best-checkpoint SHA-256: e4c5b4e5cf7dde43d670d1605910b2092ab1ed309260554a8c77ea4ce4c30eec
- Best epoch: 3
- Input preparation: 518x518 aspect-preserving bilinear resize, centered RGB letterbox with padding (124, 116, 104), then ImageNet mean/std normalization (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225).
- Descriptor: final-LayerNorm CLS token concatenated with mean of all final-LayerNorm patch tokens, 1536 dimensions.
- Encoder training delta vs public weights: 56 state tensors changed, all within transformer blocks 8–11; the other 118 encoder tensors are unchanged. This was checked against the pinned public safetensors.
- The prior-corrected and raw DINO classifiers use this same encoder state. Adding the fixed train-prior logit offset is an inference rule and does not change these tensors.

CPU loading and descriptor wrapper:

~~~python
import torch
import timm

artifact = torch.load(
    "/home/zjkj/24334067/OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_epoch3_encoder.pt",
    map_location="cpu",
    weights_only=True,
)
encoder = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0
)
encoder.load_state_dict(artifact["encoder_state_dict"], strict=True)
encoder.eval()

def encode(images):
    # images: NCHW float tensor already prepared at 518x518 and ImageNet-normalized
    tokens = encoder.forward_features(images)  # final LayerNorm output, N x 1370 x 768
    return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)  # N x 1536

with torch.inference_mode():
    features = encode(images)
~~~

Calling the raw timm encoder returns only a 768-dimensional CLS embedding. The wrapper uses forward_features to obtain all normalized tokens and builds the 1536-dimensional transfer descriptor above. A torchvision ViT-B/16 checkpoint and a 224-pixel pipeline are not directly compatible with this artifact. SYSU integration was not performed in this task.


## SAM-DINO-last4 encoder artifact

Artifact: dinov2_last4_sam_epoch16_encoder.pt

This second artifact is the headless encoder state from the DINOv2 last-four-block SAM(AdamW) checkpoint at best epoch 16. It uses SAM rho=0.05, is paired with the same pinned public model revision and 518-pixel aspect-preserving RGB letterbox / ImageNet normalization, and exports the same 1536-dimensional CLS-plus-mean-patch descriptor as the epoch-3 baseline. It is a separate encoder state, not an update of the epoch-3 artifact.

To reproduce the SAM-trained variant, point the same strict-load wrapper above to /home/zjkj/24334067/OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_sam_epoch16_encoder.pt. Use the epoch-3 artifact when reproducing the non-SAM last4 baseline. The completed fixed internal test gave SAM 236/313 (75.40%), macro-F1 .780181 and balanced accuracy .797178, versus 227/313 (72.52%) for the raw non-SAM baseline. SAM is the recommended candidate for subsequent SYSU transfer, whose effectiveness has not been evaluated. The prior-corrected classifier uses the same encoder tensors as its raw classifier; prior correction affects only the output-logit rule.

The SAM artifact SHA-256 is 58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65; source full-checkpoint SHA-256 is 2ea7daf27541571a6c2163c28d861fd3cb53998bc881eaba0288432424b9a838. Its separate audit JSON records exact equality of all 174 exported tensors with the source full-model encoder mapping, and 56 changed tensors versus the pinned public weights, limited to blocks 8–11. After export, root independently checked the actual artifact hash, strict-loaded all 174 encoder tensors into the matching architecture and reproduced the source full-model 1536-dimensional descriptor exactly on one synthetic 518-pixel CPU input. Export verification used no real images or GPU; SYSU integration was not performed.
