# FBAIs_AIproject
AI-Powered Deepfake &amp; Fake Media Detection Platform Building from the Challenge — Developing contextualized learning experiences through rigorous, content-based research to create a foundation for actionable and sustainable solutions

# Objective 1 : Fake Media Detection

Deep learning pipeline for detecting AI-generated and deepfake media
(images and videos) with explainable AI.

**Sub-objectives covered:**
-  AI-generated vs real images
-  Deepfake faces vs real faces (image-level)
-  Deepfake video detection (frame-level via MTCNN extraction)

**Approach for each task:**
- Custom CNN baseline (from scratch)
- Multiple pretrained CNN backbones with transfer learning
- XAI: Grad-CAM (+ Guided Grad-CAM and LLaVA textual explanation for video)
- Standardized benchmark (accuracy, F1, ROC-AUC, confusion matrix)
