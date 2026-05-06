# COCO Contrastive Learning — EfficientNet + DistilBERT

## Overview
Image-caption contrastive learning model trained from scratch on COCO 2017.
Developed as part of a Deep Learning university project.
Inspired by CLIP, built entirely with PyTorch and HuggingFace Transformers.

## Features
- EfficientNet-B0/B4 image encoder + DistilBERT text encoder
- InfoNCE symmetric loss with hard negative mining
- Embedding queue (FIFO, ~2MB) for extra negatives
- Gradient accumulation (effective batch 256)
- Multi-caption averaging at eval (+3-5% R@1)
- Early stopping on R@1 + safety net on val loss
- Permanent checkpoint saving via Kaggle Dataset API

## Tech Stack
### Vision
- PyTorch, TorchVision
- EfficientNet-B0 / B4 (ImageNet pretrained)

### Language
- HuggingFace Transformers
- DistilBERT-base-uncased (frozen transformer, free embeddings)

### Other Tools
- pycocotools (COCO annotations)
- Mixed precision (torch.amp)
- Kaggle Secrets + Dataset API

## Directory Structure
├── train_from_scratch()   # Full training pipeline
├── train_resume()         # Resume from any checkpoint
├── EmbeddingQueue         # FIFO negative buffer
├── ImageCaptionScorer     # Inference interface
└── plot_history()         # Training curves (loss, acc, R@K, gap)

## Getting Started
1. Mount COCO 2017 dataset on Kaggle
2. Set your Kaggle API token in Secrets
3. Run train_from_scratch() or train_resume("epoch_XX.pt")

## Acknowledgments
Inspired by CLIP (Radford et al., 2021) and MoCo (He et al., 2020).
COCO 2017 dataset — Lin et al., 2014.