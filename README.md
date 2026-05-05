# FBAIs_AIproject
AI-Powered Deepfake &amp; Fake Media Detection Platform Building from the Challenge — Developing contextualized learning experiences through rigorous, content-based research to create a foundation for actionable and sustainable solutions

# Objective 1 : Fake Media Detection

Deep learning pipeline for detecting AI-generated and deepfake media (images and videos) with explainable AI.

**Sub-objectives covered:**
-  AI-generated vs real images
-  Deepfake faces vs real faces (image-level)
-  Deepfake video detection (frame-level via MTCNN extraction)

##  Fake image detection — `real-fake-deepfake-detection.ipynb`

Two-part pipeline for detecting fake images, structured as independent tasks:

**Part 1 — AI-generated vs Real**
Dataset: `ai-generated-images-vs-real-images` (Tristan Zhang)

**Part 2 — Deepfake faces vs Real**
Dataset: `140k-real-and-fake-faces` (xhlulu)

**For each part:**
- Data exploration + balance check
- Data preparation (224×224, ImageNet-style normalization, augmentation,
  10% validation split carved out of train without modifying train/test loaders)
- Custom **BaseCNN** trained from scratch (6 epochs Part 1, 5 epochs fine-tune Part 2)
- 3 pretrained models with **2-phase fine-tuning** (head only → unfreeze last blocks):
  ResNet50, EfficientNet-B0, Xception (timm)
- **Grad-CAM** explainability tested on 2 images max (1 real + 1 fake) with true class displayed
- Test-set evaluation: accuracy, F1 macro, ROC-AUC, precision, recall,
  confusion matrix, ROC curve, learning curves
- Benchmark CSV per task

**Tech stack:** PyTorch, torchvision, timm, scikit-learn, OpenCV

## 🎬 Deepfake video — pretrained models — `deepfake-videos-detection-pretrained-models.ipynb`

Production-candidate models for deepfake video detection. Three pretrained
backbones fine-tuned on Celeb-DF v2 face frames, with multimodal explainability.

**Pipeline:**
- Dataset: Celeb-DF v2 (Celeb-real + YouTube-real + Celeb-synthesis)
- **MTCNN face extraction**: 5 frames per video, 150 videos per class,
  margin=20, GPU-accelerated
- ImageFolder dataset, 80/20 train/val split with separate transforms
  (deepcopy trick), batch size 64
- Augmentations: RandomFlip, Rotation 15°, ColorJitter 0.2

**Three pretrained models, 30 epochs each:**
- **EfficientNet-B0** — Adam, lr=1e-4
- **Xception** (via timm) — Adam, lr=1e-4
- **ResNet50** — Adam, lr=1e-4 + weight_decay=1e-4
- Full fine-tuning, FC head replaced (→ 2 classes), StepLR scheduler (step=10, γ=0.5)
- Multi-GPU support (DataParallel on 2× T4), best checkpoint saved on val_acc

**Multimodal XAI:**
- **Grad-CAM** — global region focus on last conv layer
- **Guided Grad-CAM** — pixel-level fine-grained attribution
- **LLaVA-1.5-7B** (4-bit quantized) — natural-language interpretation of heatmaps

**Evaluation:**
- Multi-frame video test (4 frames per video → mean fake probability)
- Tested on both real and fake videos for ground-truth verification
- Cybersecurity-oriented metrics: Precision, Recall, F1-Score
- Confusion matrices + learning curves comparison
- 3-way model benchmark for deployment recommendation

**Tech stack:** PyTorch, torchvision, timm, facenet-pytorch, pytorch-grad-cam,
transformers (LLaVA), bitsandbytes, scikit-learn

## 🎬 Deepfake video — from scratch baseline — `deepfake-videos-detection-from-scratch.ipynb`

Reference baseline for the deepfake video detection sub-objective.
Builds a custom CNN on extracted face frames to provide a comparison point
against the pretrained models in the companion notebook.

**Pipeline:**
- Dataset: Celeb-DF v2 (`reubensuju/celeb-df-v2`)
- **MTCNN face extraction** (facenet-pytorch) on sampled video frames
- Extracted faces saved as ImageFolder → 224×224, train/val split
- Custom **BaseCNN** trained from scratch (Conv2D + ReLU + MaxPool stack, FC head)
- Adam optimizer, CrossEntropyLoss
- Evaluation on real and fake test videos
- Serves as the reference baseline for the multi-model benchmark

**Tech stack:** PyTorch, torchvision, facenet-pytorch, OpenCV
