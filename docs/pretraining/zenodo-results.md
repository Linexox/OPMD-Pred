# Zenodo final-results audit (2026-10-07)

Reconciliation for the final report, updated with the root agent's completed SAM checkpoint, validation, test and encoder-export audits. The progress note at experiment-progress-2026-10-07.md contains a long chronology with stale intermediate statuses; use the table below as the current evidence index. This report update launched no additional training or inference.

## Protocol and interpretation

- Fixed outer split: seed 42; train/validation/test are 2366/315/313 images and 571/71/71 patients. Five train-side exact-source duplicate images were excluded to preserve the validation/test sets. Source-image hashes and patients are disjoint across the retained outer splits.
- Model inputs are full-frame oral images and supervision is Category. Patient ID is split/provenance metadata only. No SYSU data, clinical metadata, or ROI labels are used as model inputs.
- The same test set has been used for multiple exploratory candidates. Treat these results as an internal benchmark, not a newly independent clinical validation. Test OCA support is only 6.
- The current winning test result is SAM-DINO-last4: 236/313 = 75.40%, macro-F1 .780181 and balanced accuracy .797178. The >75% minimum target was reached; the optional 80% target was not reached.
- The fixed B0 + prior-corrected DINO soft vote has validation results only; it was not evaluated on test. SAM was the last new training method, as requested by the user; training and fixed-checkpoint evaluations are complete.
- The old main-branch encoder-only evaluation retained a last-epoch classifier head and is not a comparable evaluation of the best full checkpoint.

## Candidate results

Best validation metrics are from saved metrics/JSON or the selected checkpoint history. Encoder dimension is the exported headless descriptor. “Encoder update” describes training within that candidate; frozen feature/SVM and inference-only corrections do not update it.

| Method | Branch (HEAD) | Best validation | Test status/result | Encoder dim / update | Source output |
|---|---|---:|---:|---|---|
| ResNet50 ImageNet V2 full fine-tune | codex/zenodo-resnet50 (5e3fb95f) | 204/315, 64.76% | 211/313, 67.41% | 2048; full encoder updated | OPMD-Pred-Zenodo-ResNet50/outputs/zenodo_resnet50_seed42 |
| EfficientNet-B0 ImageNet V1 RAW | codex/zenodo-efficientnet-b0-480 (67370b47) | 208/315, 66.03% | 223/313, 71.25% | 1280; full encoder updated | OPMD-Pred-Zenodo-EfficientNetB0-480/outputs/zenodo_efficientnet_b0_480_seed42 |
| DenseNet169 ImageNet V1 | codex/zenodo-densenet169 (1d69672a) | 209/315, 66.35% | 215/313, 68.69% | 1664; full encoder updated | OPMD-Pred-Zenodo-DenseNet169/outputs/zenodo_densenet169_seed42_valonly |
| B0 minority augmentation | codex/zenodo-efficientnet-b0-minority-da (f21c6352) | 209/315, 66.35% | Not evaluated | 1280; full encoder updated | OPMD-Pred-Zenodo-EfficientNetB0-DA/outputs/zenodo_efficientnet_b0_480_minority_da_seed42 |
| EfficientNet-B4 RAW | codex/zenodo-efficientnet-b4-480 (56135daa) | 199/315, 63.17% | Not evaluated | 1792; full encoder updated | OPMD-Pred-Zenodo-EfficientNetB4-480/outputs/zenodo_efficientnet_b4_480_seed42 |
| WS-DAN adaptation, B0/M8 | codex/zenodo-wsdan-b0 (a2ae4f62) | 222/315, 70.48% | 198/313, 63.26% | 10240; B0 encoder and attention trained | OPMD-Pred-Zenodo-WSDAN/outputs/zenodo_wsdan_b0_seed42_valonly |
| B0 staged warmup/fine-tune | codex/zenodo-efficientnet-b0-staged (a3e06ca2) | 202/315, 64.13% | Not evaluated | 1280; full encoder updated | OPMD-Pred-Zenodo-EfficientNetB0-Staged/outputs/zenodo_efficientnet_b0_staged_seed42 |
| Fixed B0 + DenseNet soft vote | codex/zenodo-b0-densenet-softvote (fa81f22e) | 222/315, 70.48% | 220/313, 70.29% | 2944 combined; component encoders previously fine-tuned, no fusion training | OPMD-Pred-Zenodo-B0-DenseNet-Ensemble/outputs/zenodo_b0_densenet_softvote_{validation,test}.json |
| B0 Attention MIL | codex/zenodo-attention-mil-b0 (240b2755) | 210/315, 66.67% | Not evaluated | 2560; shared encoder trained | OPMD-Pred-Zenodo-AttentionMIL-B0/outputs/zenodo_attention_mil_b0_seed42 |
| ConvNeXt V2-Tiny, 1K/224 letterbox | codex/zenodo-convnextv2-tiny-224 (dbe83bfd) | 202/315, 64.13% | Not evaluated | 768; encoder fine-tuned | OPMD-Pred-Zenodo-ConvNeXtV2-Tiny/outputs/zenodo_convnextv2_tiny_224_seed42 |
| ConvNeXt V2-Tiny, 22K→1K/384 letterbox | codex/zenodo-convnextv2-tiny-384 (fbcab3a5) | 204/315, 64.76% | Not evaluated | 768; encoder fine-tuned | OPMD-Pred-Zenodo-ConvNeXtV2-Tiny-384/outputs/zenodo_convnextv2_tiny_384_seed42 |
| B0 seeds 42/43/44 soft vote | codex/zenodo-b0-seed-ensemble (02d5de55) | 221/315, 70.16% | 214/313, 68.37% | 3840 combined; three encoders trained, no learned fusion | OPMD-Pred-Zenodo-B0-Seed-Ensemble/outputs/zenodo_b0_seed_ensemble_{validation,test}.json |
| EfficientNetV2-S RAW | codex/zenodo-efficientnet-v2-s-480 (40884fd3) | 211/315, 66.98% | 205/313, 65.50% | 1280; full encoder updated | OPMD-Pred-Zenodo-EfficientNetV2S-480/outputs/zenodo_efficientnet_v2_s_480_seed42 |
| DINOv2 frozen CLS + patch-mean probe | codex/zenodo-dinov2-b14-518-logreg (3895dc01) | 182/315, 57.78% | Not evaluated | 1536; encoder frozen, logistic head only | OPMD-Pred-Zenodo-DINOv2-518-LogReg/outputs/zenodo_dinov2_b14_518_logreg_seed42 |
| DINOv2 last four blocks | codex/zenodo-dinov2-b14-518-last4 (379fda12) | 222/315, 70.48% | 227/313, 72.52% | 1536; only last four blocks plus head updated | OPMD-Pred-Zenodo-DINOv2-518-Last4/outputs/zenodo_dinov2_b14_518_last4_seed42 |
| B0 L2-SP alpha .001 | codex/zenodo-b0-l2sp-480 (7564abfa) | 212/315, 67.30% | 216/313, 69.01% | 1280; full encoder updated with anchor penalty | OPMD-Pred-Zenodo-EfficientNetB0-L2SP/outputs/zenodo_efficientnet_b0_l2sp_a001_seed42 |
| B0 L2-SP alpha .01 | codex/zenodo-b0-l2sp-480 (7564abfa) | 211/315, 66.98% | Not evaluated | 1280; full encoder updated with anchor penalty | OPMD-Pred-Zenodo-EfficientNetB0-L2SP/outputs/zenodo_efficientnet_b0_l2sp_a01_seed42 |
| B0 training weighted CE | codex/zenodo-b0-weighted-ce-480 (45e981cf) | 205/315, 65.08% | Not evaluated | 1280; full encoder updated | OPMD-Pred-Zenodo-EfficientNetB0-WeightedCE/outputs/zenodo_efficientnet_b0_weighted_ce_480_seed42 |
| Frozen adapted B0 features + SVM | codex/zenodo-b0-features-svm (ec6e2107) | 205/315, 65.08% | Not evaluated | 1280; no new encoder update, scaler/SVM fit on train | OPMD-Pred-Zenodo-B0-Features-SVM/outputs/zenodo_b0_features_svm_seed42/selection |
| DINO last4 with fixed train-prior correction | codex/zenodo-dinov2-prior-correction (fce84b61) | 224/315, 71.11% | 230/313, 73.48% | 1536; no new encoder update, inference bias only | OPMD-Pred-Zenodo-DINOv2-Prior-Correction/outputs/zenodo_dinov2_prior_correction_{validation,test}.json |
| Fixed B0 + prior-corrected DINO soft vote | codex/zenodo-b0-dino-softvote (536dfeb4) | 230/315, 73.02% | Not evaluated | 2816 combined; previously trained component encoders, no fusion training | OPMD-Pred-Zenodo-B0-DINO-Softvote/outputs/zenodo_b0_dino_softvote_validation.json |
| Final DINO-last4 SAM adaptation | codex/zenodo-dinov2-last4-sam (a3324c86) | 229/315, 72.70% | 236/313, 75.40% | 1536; last four blocks plus head updated with SAM(AdamW), best epoch 16 | OPMD-Pred-Zenodo-DINOv2-Last4-SAM/outputs/zenodo_dinov2_b14_518_last4_sam_seed42 |

The protocol branch codex/zenodo-protocol is dd8f9f84 and contains the shared split/training protocol rather than a separate candidate result.

## Focused source JSON cross-check

- RAW B0 metrics: /home/zjkj/24334067/OPMD-Pred-Zenodo-EfficientNetB0-480/outputs/zenodo_efficientnet_b0_480_seed42/metrics.json. Its saved best epoch is 5. Test confusion matrix is [[38,17,33,0],[3,62,9,0],[15,11,119,0],[2,0,0,4]].
- DINO-last4 raw metrics/test replay: /home/zjkj/24334067/OPMD-Pred-Zenodo-DINOv2-518-Last4/outputs/zenodo_dinov2_b14_518_last4_seed42/{metrics.json,test_evaluation.json}; best epoch 3 and test confusion [[45,10,33,0],[6,58,10,0],[11,15,118,1],[0,0,0,6]].
- DINO prior correction: /home/zjkj/24334067/OPMD-Pred-Zenodo-DINOv2-Prior-Correction/outputs/zenodo_dinov2_prior_correction_{validation,test}.json; same best epoch 3 and checkpoint as raw DINO; no test-derived threshold or training.
- Fixed B0-DINO vote: /home/zjkj/24334067/OPMD-Pred-Zenodo-B0-DINO-Softvote/outputs/zenodo_b0_dino_softvote_validation.json. Bundle SHA-256 35bb59af59b46cfddd498cd8a1633b1a2781a2eaf773d844c8f051062bd3f65d; test not run. Its initial manifest-key error happened before model construction; the retry used commit c0093d0901fbbe42ba0ecb60c7c57034034d00a5 and was independently checked.
- Final SAM completed 20 epochs; validation selected epoch 16. Source files: /home/zjkj/24334067/OPMD-Pred-Zenodo-DINOv2-Last4-SAM/outputs/zenodo_dinov2_b14_518_last4_sam_seed42/{metrics.json,validation_evaluation.json,test_evaluation.json}. Root read the completed files and independently recomputed checkpoint SHA-256 `2ea7daf27541571a6c2163c28d861fd3cb53998bc881eaba0288432424b9a838` and manifest SHA-256 `8287b1d5a2338836b9ce9b9f62e4e0b7dfde467b9bef1d6afdcb13ab1660845a`. Validation replay matched all epoch-16 metrics, loss and confusion matrix. One fixed raw test used the same model and manifest: confusion matrix [[46,7,34,1],[7,60,7,0],[9,12,124,0],[0,0,0,6]], accuracy 236/313, macro-F1 .780180995, balanced accuracy .797177624. No test-derived adjustments were made.
- SAM headless export: /home/zjkj/24334067/OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_sam_epoch16_encoder.pt; SHA-256 `58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65`. Root verified 174 tensors exactly match the full source encoder and contain no head; strict-loading into the corresponding DINO model reproduced the full-model 1536-dimensional descriptor on a synthetic CPU input. No SYSU evaluation was performed.

## Synced branch references

These 22 local codex/zenodo refs were reconciled from the synced Git bundle refs. Short SHAs are the current local branch tips for identifying the source code snapshots; they are not PR or remote-push status.

| Branch | HEAD |
|---|---|
| codex/zenodo-protocol | dd8f9f84 |
| codex/zenodo-resnet50 | 5e3fb95f |
| codex/zenodo-efficientnet-b0-480 | 67370b47 |
| codex/zenodo-densenet169 | 1d69672a |
| codex/zenodo-efficientnet-b0-minority-da | f21c6352 |
| codex/zenodo-efficientnet-b4-480 | 56135daa |
| codex/zenodo-wsdan-b0 | a2ae4f62 |
| codex/zenodo-efficientnet-b0-staged | a3e06ca2 |
| codex/zenodo-b0-densenet-softvote | fa81f22e |
| codex/zenodo-attention-mil-b0 | 240b2755 |
| codex/zenodo-convnextv2-tiny-224 | dbe83bfd |
| codex/zenodo-convnextv2-tiny-384 | fbcab3a5 |
| codex/zenodo-b0-seed-ensemble | 02d5de55 |
| codex/zenodo-efficientnet-v2-s-480 | 40884fd3 |
| codex/zenodo-dinov2-b14-518-logreg | 3895dc01 |
| codex/zenodo-dinov2-b14-518-last4 | 379fda12 |
| codex/zenodo-b0-l2sp-480 | 7564abfa |
| codex/zenodo-b0-weighted-ce-480 | 45e981cf |
| codex/zenodo-b0-features-svm | ec6e2107 |
| codex/zenodo-dinov2-prior-correction | fce84b61 |
| codex/zenodo-b0-dino-softvote | 536dfeb4 |
| codex/zenodo-dinov2-last4-sam | a3324c86 |
