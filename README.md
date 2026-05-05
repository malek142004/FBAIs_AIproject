# FBAIs_AIproject
AI-Powered Deepfake &amp; Fake Media Detection Platform Building from the Challenge — Developing contextualized learning experiences through rigorous, content-based research to create a foundation for actionable and sustainable solutions

# Objective 2 : Image Tampering Detection

Deep learning pipeline for detecting and localizing image manipulation on the DF2023 dataset, with dual-output models (binary segmentation mask + 4-class classifier) and explainable AI.

**Sub-objectives covered:**
- 4-class manipulation type classification: Inpainting (I), Copy-move (C), Splicing (S), Enhancement (E)
- Pixel-level tampered region localization (binary segmentation mask)
- From-scratch baseline vs. frozen pretrained encoders (ConvNets + Transformers)

## Image tampering baseline — `cnn-scratch-tampering.ipynb`

From-scratch dual-output CNN serving as the required professor baseline for the DF2023 sub-objective. Expected to under-perform the pretrained variants — that comparison is the point.

**Dataset:** DF2023 V15 (`hiwotyirga/the-digital-forensics-2023-dataset-df2023`) — 40k/class balanced subset (160k total), 70/15/15 train/val/test split, 256×256

**Architecture:**
```
Input [B, 3, 256, 256]
        ↓
Encoder (4 DoubleConv stages + MaxPool): 3→32→64→128→256
        ↓
Bottleneck [B, 256, 16, 16]
   ↙                          ↘
U-Net Decoder (4 upsamples    Classification head
+ skip concatenations)        (AvgPool → Dropout → Linear)
        ↓                              ↓
Binary mask [B, 1, 256, 256]    4-class logits [B, 4]
        ↓                              ↓
   Dice + BCE                     CrossEntropy
         ↘                       ↙
       total = 0.8·seg + 0.2·cls
```
- 2.15M trainable parameters, 12.80 GFLOPs, 208 FPS @ batch 64

**Training:** AdamW lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, AMP (automatic mixed precision), early stopping (patience=2, min_delta=1e-4), 2×T4 DataParallel, up to 10 epochs. Best checkpoint saved on val loss.

**Explainability (xAI):**
- **Grad-CAM** on `enc4` (last encoder stage) — shows where the classifier focuses
- **Vanilla saliency maps** — pixel-level `|∂score/∂x|` attribution
- **Qualitative panels** — image · GT mask · predicted probability · overlay (one sample per class)
- **t-SNE** of 256-d bottleneck features (PCA-50 pre-reduction, 1500 test samples)
- **Calibration analysis** — reliability diagram + Expected Calibration Error (ECE)
- Model diagnostics: per-module parameter breakdown, conv-FLOPs, GPU latency/FPS

**Test-set evaluation:** accuracy, F1 (macro/weighted), ROC-AUC (OvR), per-class precision/recall/F1, mean IoU, mean Dice, per-class IoU/Dice, raw + row-normalised confusion matrices

| Metric | Value |
|---|---|
| Test Accuracy | 0.9520 |
| Test F1 (macro) | 0.9522 |
| Test ROC-AUC (macro/OvR) | 0.9959 |
| Test mean IoU | 0.7065 |
| Test mean Dice | 0.7790 |
| ECE | 0.0139 |
| Throughput | 208 FPS @ batch 64 |

**Tech stack:** PyTorch, torchvision, scikit-learn, OpenCV, matplotlib, seaborn

## Image tampering — ConvNeXt-V2 Tiny — `convnex-v2-tiny.ipynb`

ConvNeXt-V2 Tiny (FCMAE → ImageNet-22k → ImageNet-1k pretrained, **fully frozen encoder**) with a U-Net decoder and classification head. Architectural counterpoint to the from-scratch CNN — quality difference is attributable to features, not extra trainable capacity.

**Dataset:** DF2023 V15 — 15k/class balanced subset (60k total), 70/15/15 split, pre-resized to 256×256 on local SSD (avoids FUSE-mount latency on Kaggle), batch 96

**Architecture:**
```
Input [B, 3, 256, 256]
        ↓
ConvNeXt-V2 Tiny encoder (FULLY FROZEN, 27.86M params)
   stem + stages[0..3] → features at strides 4, 8, 16, 32
   channels: [96, 192, 384, 768]
        ↓
Bottleneck [B, 768, 8, 8]
   ↙                          ↘
U-Net decoder                  Classification head
(4 upsamples w/ skips)         (AvgPool → Dropout → Linear)
        ↓                              ↓
Binary mask [B, 1, 256, 256]    4-class logits [B, 4]
```
- 31.97M total | 4.11M trainable (12.84%) | frozen: 27.86M
- GPU-side normalization: uint8 tensors transferred, cast + normalize on device

**Training:** AdamW lr=1e-3 (decoder + head only), weight_decay=1e-4, CosineAnnealingLR, AMP, early stopping (patience=2), single-GPU T4 (DataParallel excluded — overhead exceeds benefit with frozen encoder), up to 10 epochs

**Explainability (xAI):**
- **Grad-CAM** on `stages[-1]` (deepest stage, 8×8 spatial, 768 channels) — gradients flow through frozen encoder at inference
- **Vanilla saliency maps**
- **t-SNE** of 768-d GAP'd bottleneck features (PCA-50 → 2D)
- Calibration + ECE, full benchmark suite (same as baseline)

**Tech stack:** PyTorch, timm (`convnextv2_tiny.fcmae_ft_in22k_in1k`), scikit-learn, OpenCV

## Image tampering — DINOv2 ViT-S/14 — `dinov2-vit.ipynb`

Self-supervised DINOv2 ViT-S/14 (Meta AI, **fully frozen**) with a lightweight CNN mask decoder and a linear classifier — tests whether label-free pretraining on 142M images encodes tampering-relevant structure without supervised fine-tuning.

**Dataset:** DF2023 V15 — 15k/class balanced subset, input size **224×224** (DINOv2 uses 14px patches → 16×16 token grid at this resolution), batch 128

**Architecture:**
```
Input [B, 3, 224, 224]
        ↓
DINOv2 ViT-S/14 (FULLY FROZEN, 22.06M params)
   patch embed → 12 transformer blocks
   → CLS token  [B, 384]           (classification)
   → patch tokens [B, 256, 384]    (mask decoder)
   ↙                               ↘
patch grid → conv decoder           Linear classifier
(384→192→96→48→24 + bilinear ups)  (Dropout → Linear 384→4)
        ↓                                   ↓
Binary mask [B, 1, 224, 224]         4-class logits [B, 4]
```
- 22.94M total | 0.88M trainable (3.85%) — smallest trainable footprint in the project

**Training:** AdamW lr=1e-3 (decoder + classifier only), CosineAnnealingLR, AMP, early stopping (patience=2), single-GPU T4, up to 10 epochs. Requires `xformers` for efficient attention.

**Explainability (xAI):**
- **Grad-CAM** on the last ViT transformer block — token sequence `[B, N+1, D]` reshaped to 16×16 spatial grid (CLS token stripped) before gradient-weighted sum
- **Vanilla saliency maps**
- **t-SNE of CLS-token embeddings** — directly probes whether frozen self-supervised features separate the four manipulation classes without any supervision
- Calibration + ECE, full benchmark suite

**Tech stack:** PyTorch, xformers, `torch.hub` (`facebookresearch/dinov2`), scikit-learn, OpenCV

## Image tampering — SegFormer MiT-B0 — `segformer-mit-b0.ipynb`

SegFormer MiT-B0 hierarchical transformer (ImageNet pretrained, **fully frozen**) paired with an `smp` U-Net decoder — the canonical "transformer for dense prediction" baseline and architectural counterpoint to all ConvNet notebooks.

**Dataset:** DF2023 V15 — 15k/class balanced subset (60k total), 70/15/15 split, pre-resized to 256×256, batch 128

**Architecture:**
```
Input [B, 3, 256, 256]
        ↓
MiT-B0 encoder (FULLY FROZEN, 3.32M params)
   4 stages, overlap-patch embed + efficient attention O(N²/R²)
   strides 4/8/16/32, channels [32, 64, 160, 256]
        ↓
Bottleneck [B, 256, 8, 8]
   ↙                               ↘
smp U-Net decoder                   Classification head
(skip-fed, 2.23M trainable)         (AvgPool → Dropout → Linear)
        ↓                                   ↓
Binary mask [B, 1, 256, 256]         4-class logits [B, 4]
```
- 5.55M total | 2.23M trainable (40.20%) — lightest total footprint among pretrained models
- Built with `smp.Unet(encoder_name='mit_b0', aux_params=...)` — dual-output topology in one call

**Training:** AdamW lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, AMP, early stopping (patience=2), single-GPU T4 (DataParallel excluded — same reasoning as ConvNeXt notebook), up to 10 epochs

**Explainability (xAI):**
- **Grad-CAM** on MiT-B0 encoder (returns list of feature maps — hook picks the deepest, stride-32, 8×8)
- **Vanilla saliency maps**
- **t-SNE** of 256-d bottleneck features (GAP of deepest encoder stage, PCA-50 → 2D)
- Calibration + ECE, full benchmark suite

**Tech stack:** PyTorch, segmentation-models-pytorch (`smp`), scikit-learn, OpenCV
