# DINOv2-B/14 last-four-block fine-tuning with SAM

## Scope and prior evidence

This branch changes only the minibatch optimizer update for the existing full-frame, patient-level four-class DINOv2 last-four recipe. It starts from the same pinned public DINOv2 initialization as that recipe, not from a trained Zenodo checkpoint. Model inputs remain full RGB oral images and the only supervision is the four-class Category label. Patient identifiers are used only for split construction; no ROI annotations, diagnosis, demographics, or hospital metadata enter the model.

The motivation is to test whether flatter-neighborhood optimization reduces the observed train/validation gap without changing the representation architecture or training objective. Foret et al. describe SAM as minimizing loss in neighborhoods around the current weights [original paper](https://arxiv.org/abs/2010.01412). Alhaj et al. report small average validation-accuracy gains from SAM on BUSI and BUS-UCLM ultrasound datasets in a 2026 medical transfer-learning study ([Scientific Reports paper](https://www.nature.com/articles/s41598-026-49376-4)). The study randomly partitioned images 70/30 and did not group by patient, so those results are not patient-held-out evidence and do not predict Zenodo performance. Its author code also has a shared-transform issue; this implementation follows the SAM update formula rather than claiming exact reproduction.

## Fixed model and data recipe

- `timm` `vit_base_patch14_dinov2.lvd142m`, pinned revision `4685c99dabffe5affac90bd99dbffd25801ae58d`; pretrained `model.safetensors` SHA-256 `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b`.
- Train only ViT blocks 8–11 and the new 1536→4 linear head. Keep patch embedding, tokens, blocks 0–7, and final LayerNorm frozen. The descriptor is final-LN CLS concatenated with the mean of all final-LN patch tokens.
- Use the audited uint8 518×518 RGB full-frame letterbox cache with bilinear aspect-preserving resize and centered `(124,116,104)` padding. Training retains the existing horizontal-flip and mild ColorJitter transform; validation and test are unaugmented.
- Keep patient/source-JPEG-hash split seed 42, train 2366, validation 315, test 313; keep training seed 42. Train-count inverse-frequency weighted cross-entropy remains unchanged.
- Keep AdamW encoder LR `1e-5`, head LR `1e-3`, weight decay `1e-4`, batch 8, up to 20 epochs, patience 5, validation-accuracy checkpointing with macro-F1 tie-break. Test is not used for model or setting selection.

## SAM update

Use ordinary (non-adaptive) SAM with fixed `rho=0.05`; do not tune rho on validation or test. For the first forward/backward pass on the already augmented minibatch, calculate the weighted CE gradient `g` and its single global L2 norm over all trainable last-four-block and head parameters. Form the perturbation `epsilon = rho * g / (||g||2 + 1e-12)` and temporarily add it only to parameters that received first-pass gradients. No optimizer step, Adam state update, or weight decay occurs at this perturbed point.

Clear gradients, forward/backward the same image tensor and labels at `w + epsilon`, and then restore every perturbed parameter from an exact saved clone inside `finally`. Apply the existing global gradient clipping threshold 1.0 to this second-pass gradient, then take one AdamW step from the restored original weights. Thus AdamW moments and decoupled weight decay advance only once per batch. The first-pass loss and predictions are used for train metrics, so their meaning remains comparable to the baseline; validation uses one ordinary forward pass. The ViT uses LayerNorm and has no BatchNorm running statistics to double-update.

The training args/checkpoint/metrics record `SAM(AdamW)`, `rho`, `adaptive=False`, two forward/backward passes per minibatch, and that clipping applies to the second-pass gradient. The saved model/checkpoint structure, encoder export, features, classifier, and inference cost are unchanged. This is an optimizer adaptation, not an exact reproduction of a medical ultrasound paper. SAM roughly adds one forward/backward pass per training batch and has no extra inference pass. It does not guarantee accuracy above 0.75 or 0.80.

## Environment and implementation checks

The CPU reference check verifies the perturbation norm, exact parameter restoration, frozen-parameter invariance, second-gradient AdamW update, and that AdamW state advances once. The completed run used the remote personal virtual environment: Python 3.14.7, PyTorch 2.14.0, torchvision 0.29.0, `timm` 1.0.29, `huggingface-hub` 1.32.0, and `safetensors` 0.8.0. In a fresh environment, install compatible PyTorch/torchvision builds for the target CUDA runtime, then install `timm==1.0.29 huggingface-hub==1.32.0 safetensors==0.8.0`; these packages are not declared in the project dependency file. The run used the existing personal venv, `PYTHONPATH=src`, and the pinned offline DINOv2 weight snapshot.

## Completed training and evaluation

The fixed run completed 20 epochs with seed 42 and split seed 42. Best checkpoint selection used validation accuracy with macro-F1 as the tie-break; epoch 16 was selected. The trainer evaluates train and validation only. The test split was evaluated once afterward with the saved epoch-16 checkpoint and the same fixed patient/source-hash manifest. No test result was used to select a checkpoint or change the recipe.

Training artifacts are in `outputs/zenodo_dinov2_b14_518_last4_sam_seed42/`: `checkpoint.pt`, `metrics.json`, `args.json`, `split_manifest.json`, and `train.log`. Checkpoint SHA-256 is `2ea7daf27541571a6c2163c28d861fd3cb53998bc881eaba0288432424b9a838`; split-manifest SHA-256 is `8287b1d5a2338836b9ce9b9f62e4e0b7dfde467b9bef1d6afdcb13ab1660845a`. The independently replayed validation JSON is `outputs/zenodo_dinov2_b14_518_last4_sam_seed42/validation_evaluation.json`; every classification metric, confusion matrix, and loss matched epoch 16 history.

| Split | N | Accuracy | Macro-F1 | Balanced accuracy | Loss |
|---|---:|---:|---:|---:|---:|
| Validation | 315 | 0.72698 | 0.71400 | 0.71867 | 0.99352 |
| Test | 313 | 0.75399 (236/313) | 0.78018 | 0.79718 | 0.92886 |

Test support follows the fixed order Benign, Healthy, OPMD, OCA: 88, 74, 145, 6. Per-class recall is 0.52273, 0.81081, 0.85517, and 1.00000. The test confusion matrix in the same order is:

```text
[[46,  7, 34, 1],
 [ 7, 60,  7, 0],
 [ 9, 12,124, 0],
 [ 0,  0,  0, 6]]
```

The exact training command was:

```bash
cd /home/zjkj/24334067/OPMD-Pred-Zenodo-DINOv2-Last4-SAM
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 \
  /home/zjkj/24334067/OPMD-Pred/.venv/bin/python -m opmd_pred.scripts.train_dinov2_last4 \
  --csv /home/zjkj/24334067/OPMD-Pred/data/zenodo/Imagewise_Data.csv \
  --images /home/zjkj/24334067/OPMD-Pred/data/zenodo/Images \
  --cache /home/zjkj/24334067/zenodo-cache/dinov2b14_letterbox_518_uint8.pt \
  --output outputs/zenodo_dinov2_b14_518_last4_sam_seed42 \
  --epochs 20 --batch-size 8 --patience 5 --encoder-lr 1e-5 --head-lr 1e-3 \
  --weight-decay 1e-4 --grad-clip 1.0 --num-workers 8 \
  --seed 42 --split-seed 42 --device cuda
```

The one-time test result is saved at `outputs/zenodo_dinov2_b14_518_last4_sam_seed42/test_evaluation.json`; command, PID, and console log are retained alongside the method output under `outputs/`.
