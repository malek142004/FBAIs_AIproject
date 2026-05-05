# 🧠 Media Credibility — Détection de Manipulation des Médias
### CBL Project | ESPRIT School of Engineering | 3rd Year AI

> **Module :** Détection de Manipulation des Médias  
> **Big Idea :** Media Credibility  
> **Étudiante :** Malek Tirellil  

---

## 🎯 Objectif du projet

Ce projet développe un **pipeline multimodal de détection de manipulation médiatique** combinant vision par ordinateur, traitement du langage naturel et un modèle de fusion from scratch.
Image publicitaire
↓
SO1-A  TrOCR      →  Extraire le texte
SO1-B  RoBERTa    →  Classifier le texte (manipulateur / neutre)
SO2    ViT-B/16   →  Détecter le clickbait visuel
SO4    DETR       →  Localiser les faux éléments d'urgence
↓
ManipNet (from scratch) → Score global de manipulation [0.0 → 1.0]
---

## 📊 Notebooks — Description

### 🔍 SO1-A — Détection de Texte avec TrOCR
**Dataset :** COCO 2014 (83K images)  
**Tâche :** Extraire le texte présent dans les images publicitaires  

| Modèle | Architecture | Taux détection | Temps/img |
|--------|-------------|----------------|-----------|
| **TrOCR ✅** | ViT Encoder + GPT-2 Decoder | **100%** | 0.95s |
| Tesseract LSTM | LSTM standalone | 95% | 0.3s |
| CRNN (EasyOCR) | CNN + BiLSTM + CTC | 20% | 0.2s |

**XAI :** Attention Map · Occlusion Sensitivity · Gradient Saliency

---

### 📝 SO1-B — Classification de Texte avec RoBERTa
**Dataset :** Clickbait Dataset Webis 2017 (32K titres, 50/50)  
**Tâche :** Classifier un texte comme manipulateur (1) ou neutre (0)  

| Modèle | F1-Score | Recall | AUC-ROC |
|--------|----------|--------|---------|
| **RoBERTa ✅** | **0.9978** | **1.0000** | **0.9999** |
| BERT | 0.9921 | 0.9934 | 0.9987 |
| DistilBERT | 0.9876 | 0.9912 | 0.9965 |

**XAI :** Attention Visualization · Integrated Gradients · LIME

---

### 🖼️ SO2 — Détection de Clickbait Visuel avec ViT-B/16
**Dataset :** MultiBanFakeDetect (Bangla Fake News, ~7 680 images)  
**Tâche :** Classifier une image comme Fake/Clickbait (1) ou Réel (0)  

| Modèle | Params | Accuracy | F1 |
|--------|--------|----------|----|
| **ViT-B/16 ✅** | 86M | **70%+** | Élevé |
| EfficientNet-B4 | 19M | Moyen | Moyen |
| ResNet-50 | 25.6M | Faible | Faible |

**Techniques anti-overfitting :** Differential LR · Mixup · Label Smoothing 0.1 · CosineWarmRestarts · WeightedSampler · RandomErasing · Early Stopping  
**XAI :** Attention Rollout · Occlusion Sensitivity · Gradient Saliency

---

### 🚨 SO4 — Détection d'Éléments d'Urgence avec DETR
**Dataset :** UI-Elements-Detection (format YOLO)  
**Tâche :** Localiser les faux éléments d'urgence (timers, badges, popups)  

| Modèle | mAP@0.5 | Precision | Recall |
|--------|---------|-----------|--------|
| **DETR ✅** | **0.2923** | **0.2690** | **0.3199** |
| YOLOv8 | 0.2263 | 0.2100 | 0.2850 |
| Faster R-CNN | 0.1424 | 0.1380 | 0.1820 |

**XAI :** Attention Encoder · Occlusion Sensitivity · Gradient Saliency

---

### 🧠 ManipNet — Modèle Multimodal From Scratch
**Dataset :** MultiBanFakeDetect  
**Tâche :** Produire un score global de manipulation [0.0 → 1.0]
Image (128×128)       → Branch CNN (4×Conv2D+BN+ReLU+GAP) → 256-dim
Texte (32 tokens)     → Branch Transformer (2 blocs Encoder) → 256-dim
Meta  (6 stats RGB)   → Branch MLP (FC 6→32→64)            →  64-dim
↓
Cross-Modal Self-Attention (4 têtes)
↓
MLP Classifier 576→256→128→64→1
↓
Score manipulation [0.0 → 1.0]
**Paramètres :** ~2-4M (100% from scratch, aucun poids pré-entraîné)  
**XAI :** Attention Modale · Gradient Saliency · Occlusion Sensitivity

---

## 🔬 Méthodes XAI utilisées

| Méthode | Notebooks | Ce qu'elle montre |
|---------|-----------|-------------------|
| Attention Map / Rollout | SO1-A, SO1-B, SO2 | Zones/tokens regardés par le modèle |
| Integrated Gradients | SO1-B | Contribution de chaque token (50 étapes) |
| LIME | SO1-B | Mots qui causent la décision (régression locale) |
| Gradient Saliency | SO1-A, SO2, SO4, ManipNet | Pixels les plus influents |
| Occlusion Sensitivity | Tous | Zones critiques (grille 6×6 ou 7×7) |
| Attention Modale | ManipNet | Quelle branche CNN/Transformer/MLP est écoutée |

---

## ⚙️ Installation

```bash
git clone https://github.com/ton-username/media-credibility.git
cd media-credibility
pip install torch torchvision transformers datasets scikit-learn \
            matplotlib seaborn opencv-python Pillow tqdm pycocotools
```

---

## 🚀 Utilisation sur Kaggle

Chaque notebook est conçu pour tourner sur **Kaggle** avec GPU T4.

### Datasets requis (à ajouter via Kaggle Datasets)
SO1-A  → jeffaudi/coco-2014-dataset-for-yolov3
SO1-B  → amananandrai/clickbait-dataset
SO2    → mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news
SO4    → daudaudinang/ui-elements-detection-dataset
ManipNet → mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news
### Chemins dans les notebooks
```python
# SO1-A
DATASET_PATH = '/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014'

# SO1-B
DATA_PATH = '/kaggle/input/datasets/amananandrai/clickbait-dataset/clickbait_data.csv'

# SO2 & ManipNet
BASE_DIR = Path('/kaggle/input/datasets/mukaffimoin/multibanfakedetect-multimodal-bangla-fake-news')

# SO4
BASE_DIR = Path('/kaggle/input/datasets/daudaudinang/ui-elements-detection-dataset')
```

---

## 📈 Résultats et performances

| Module | Modèle | Métrique principale | Valeur |
|--------|--------|---------------------|--------|
| SO1-A | TrOCR | Taux détection | 100% |
| SO1-B | RoBERTa | F1-Score | 0.9978 |
| SO1-B | RoBERTa | Recall | 1.0000 |
| SO2 | ViT-B/16 | Accuracy | 70%+ |
| SO4 | DETR | mAP@0.5 | 0.2923 |
| ManipNet | From Scratch | En cours | — |

---

## 🔧 Challenges résolus

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Overfitting SO2 (gap +32%) | 86M params sur 7 680 imgs | Freeze blocs 0-7 · Mixup · Label Smoothing |
| Data leakage ManipNet (F1=1.0) | Meta features construites depuis le label | Stats visuelles réelles (brightness, RGB) |
| TrOCR attention reshape 577→576 | Token [CLS] inclus dans la séquence | `attn_map[1:]` avant reshape |
| RoBERTa sdpa → output_attentions | Mode sdpa par défaut PyTorch ≥ 2.0 | `attn_implementation='eager'` |
| DETR tuple AttributeError | `self_attention` retourne (output, weights) | `output[0] if isinstance(output, tuple)` |

---

## 📚 Survey Paper

Un survey paper académique au format **IEEEtran** accompagne ce projet :

> **Multimodal Media Credibility: A Survey on AI-Based Detection of Fake and Manipulated Visual Content**  
> Malek Tirellil, Youssef Jouini, Yassmine Nouisser, Islem Tellili, Maryem Ouichka, Mohamed Rayen Ayat  
> ESPRIT School of Engineering, Tunisia, 2026

Sections : Introduction · Background (terminologies + GANs + Diffusion) · Detection Techniques · XAI · Evaluation Metrics · Challenges · Future Work

---

## 🗺️ TODO — Prochaines étapes

- [ ] Déploiement FastAPI + React.js sur Hugging Face Spaces
- [ ] XLM-RoBERTa pour support multilingue (FR/AR/BN)
- [ ] Brancher les vrais scores SO1-B et SO4 dans ManipNet
- [ ] Quantization INT8 pour inférence temps réel
- [ ] Dataset de pubs manipulatrices annoté spécifique
- [ ] Docker + CI/CD GitHub Actions

---

## 🏗️ Architecture du pipeline de déploiement
POST /analyze
↓
Image → TrOCR      → texte extrait
→ RoBERTa    → score_nlp [0-1]
→ ViT-B/16   → score_clickbait [0-1]
→ DETR       → bboxes urgence
→ ManipNet   → score_global [0-1]
↓
JSON { score, label, text, boxes, heatmap }
---

## 👥 Auteurs

| Nom | Spécialité |
|-----|-----------|
| Malek Tirellil | Artificial Intelligence |


**ESPRIT School of Engineering — Tunisia — 3rd Year AI — 2026**

---

## 📄 Licence

Ce projet est réalisé dans le cadre académique du programme CBL d'ESPRIT School of Engineering.

---

*"The goal is not to win a benchmark competition but to preserve the evidentiary value of visual media in an era of unprecedented synthetic capability."*  
— Survey Paper, Media Credibility Project
