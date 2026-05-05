---
name: computer-vision
description: Computer vision workflows — image data characterization, preprocessing and augmentation, architecture selection (CNN vs ViT), and evaluation metrics (mAP, IoU, FID, SSIM). Use when working with image or video data.
allowed_agents: [data, experiment]
---

# Computer Vision

## Overview

Computer vision workflows require careful attention at every stage: understanding dataset characteristics first, building a sound preprocessing and augmentation pipeline, selecting an architecture matched to dataset size and task, and evaluating with task-appropriate metrics. This skill covers the full pipeline from raw images to model evaluation.

## When to Use This Skill

Use this skill when:
- Working with image or video datasets (classification, detection, segmentation, generation)
- Designing or debugging a preprocessing/augmentation pipeline
- Selecting a model architecture for a vision task
- Computing vision-specific metrics (mAP, IoU, FID, SSIM, LPIPS)
- Transfer learning decisions (freeze vs. fine-tune, learning rate schedule)

Run the EDA skill first to understand file formats, directory structure, and basic counts. Use this skill for vision-specific analysis.

---

## Image Dataset Characterization

Before writing any training code, profile your dataset thoroughly.

```python
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2

def characterize_dataset(image_dir, extensions=('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
    image_paths = [p for p in Path(image_dir).rglob('*') if p.suffix.lower() in extensions]
    print(f"Total images: {len(image_paths)}")

    widths, heights, channels_list, aspect_ratios = [], [], [], []
    channel_means, channel_stds = [], []

    for path in image_paths:
        with Image.open(path) as img:
            w, h = img.size
            c = len(img.getbands())
            widths.append(w)
            heights.append(h)
            channels_list.append(c)
            aspect_ratios.append(w / h)

            # Per-image channel stats (sample every Nth image to stay fast)
            if len(channel_means) < 500:
                arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
                channel_means.append(arr.mean(axis=(0,1)))
                channel_stds.append(arr.std(axis=(0,1)))

    print(f"\nWidth  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
    print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
    print(f"Aspect ratio — min: {min(aspect_ratios):.2f}, max: {max(aspect_ratios):.2f}, "
          f"mean: {np.mean(aspect_ratios):.2f}")
    print(f"Channels: {Counter(channels_list)}")

    means = np.array(channel_means).mean(axis=0)
    stds  = np.array(channel_stds).mean(axis=0)
    print(f"\nChannel means (RGB): {means.round(4)}")
    print(f"Channel stds  (RGB): {stds.round(4)}")

    # Class distribution (assumes ImageFolder structure: dir/class/image.jpg)
    classes = [p.parent.name for p in image_paths]
    class_counts = Counter(classes)
    print(f"\nClass distribution ({len(class_counts)} classes):")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")

    # Plot size scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(widths, heights, alpha=0.2, s=5)
    axes[0].set_xlabel('Width'); axes[0].set_ylabel('Height')
    axes[0].set_title('Image size distribution')
    axes[1].hist(aspect_ratios, bins=50)
    axes[1].set_xlabel('Aspect ratio (W/H)'); axes[1].set_title('Aspect ratio distribution')
    plt.tight_layout(); plt.savefig('dataset_profile.png', dpi=120)

    return {'widths': widths, 'heights': heights, 'means': means, 'stds': stds}
```

Key things to flag:
- **Highly variable sizes**: need a consistent resize strategy
- **Extreme aspect ratios**: letterbox or tile instead of naive resize
- **Class imbalance >5:1**: use weighted sampling or focal loss
- **Few channels** (grayscale dataset fed to RGB model): replicate channel

---

## Preprocessing Pipeline

### Resize Strategy by Task

```python
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

# --- Classification: resize then center crop ---
transform_cls = T.Compose([
    T.Resize(256),          # Shorter side to 256
    T.CenterCrop(224),      # Crop to model input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Detection: letterbox (maintain aspect ratio, pad) ---
def letterbox(img, target_size=640, fill=(114, 114, 114)):
    """Resize with padding — preserves bounding box coordinates."""
    from PIL import Image
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new('RGB', (target_size, target_size), fill)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    new_img.paste(img, (pad_x, pad_y))
    return new_img, scale, pad_x, pad_y

# --- Segmentation: resize shortest side, then random crop ---
transform_seg = T.Compose([
    T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(512),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Normalization

```python
# ImageNet stats — use when:
#   - Fine-tuning a pretrained ImageNet model
#   - Dataset is natural images (photos, not microscopy, X-rays, etc.)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Dataset-specific stats — use when:
#   - Training from scratch
#   - Domain-specific images (medical, satellite, microscopy)
#   - ImageNet stats produce bad convergence
def compute_dataset_stats(loader):
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0
    for imgs, _ in loader:
        # imgs: [B, C, H, W], values in [0,1]
        mean += imgs.mean(dim=[0, 2, 3]) * imgs.shape[0]
        std  += imgs.std(dim=[0, 2, 3])  * imgs.shape[0]
        n    += imgs.shape[0]
    return (mean / n).tolist(), (std / n).tolist()

normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)
```

### Color Space Considerations

```python
import cv2

# OpenCV reads BGR — always convert to RGB before passing to PyTorch
img_bgr = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Grayscale — replicate to 3 channels for RGB-pretrained models
gray = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# HSV — useful for color-based segmentation (e.g., plant health, traffic signs)
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
# H: 0–179, S: 0–255, V: 0–255 in OpenCV
```

---

## Augmentation Strategies

### By Task

**Safe augmentations** (valid for almost all tasks): horizontal flip, random crop, color jitter (brightness/contrast/saturation), Gaussian blur.

**Risky augmentations** — check task-specific constraints:
- Heavy rotation: wrong if orientation is discriminative (e.g., text OCR, medical "up" orientation)
- Aggressive color jitter: wrong if color is diagnostic (e.g., plant disease, pathology staining)
- Vertical flip: wrong for aerial/satellite imagery with meaningful up/down

```python
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Classification (torchvision) ---
train_transforms_cls = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Detection (albumentations — transforms bboxes automatically) ---
train_transforms_det = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0)),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- Segmentation (albumentations — transforms mask simultaneously) ---
train_transforms_seg = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### Advanced Augmentations

```python
# Mixup — blend two images and their labels linearly
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix — paste a patch from one image into another
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam

# AutoAugment and AugMix via torchvision
auto_aug   = T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)
augmix     = T.AugMix()
rand_aug   = T.RandAugment(num_ops=2, magnitude=9)
trivial_aug = T.TrivialAugmentWide()  # strong, simple, no tuning needed
```

---

## Architecture Selection

### Decision Guide

| Scenario | Recommendation | Library |
|----------|----------------|---------|
| Small dataset (<10k images) | Fine-tune ResNet50 or EfficientNet-B3 (pretrained on ImageNet) | `torchvision.models` |
| Medium dataset (10k–100k) | ViT-S/16 or ViT-B/16 with pretrained weights | `timm` |
| Large dataset (>100k) | ConvNeXt-Base or ViT-B pretrained, fine-tune all | `timm` |
| Speed-critical inference | MobileNetV3-Large, EfficientNet-B0 | `torchvision.models` |
| Best accuracy, no speed constraint | ConvNeXt-XL, ViT-L, CLIP ViT-L/14 | `timm`, OpenAI |
| Object detection | YOLOv8 (ultralytics), DINO, Faster R-CNN | `ultralytics`, `torchvision` |
| Semantic segmentation | SegFormer, DeepLabV3+ | `transformers`, `torchvision` |

```python
import timm
import torchvision.models as models

# Pretrained ResNet50 — fine-tune for custom classification
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # replace head

# timm — massive model zoo with pretrained weights
model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)

# List available models
timm.list_models('efficientnet*', pretrained=True)[:10]
```

---

## Transfer Learning Guide

### Freeze Backbone vs Fine-Tune All

| Situation | Strategy |
|-----------|----------|
| Very small dataset (<1k images), similar to ImageNet | Freeze backbone, train head only |
| Small dataset, different domain (medical, satellite) | Freeze early layers, fine-tune last 2–3 blocks + head |
| Medium+ dataset | Fine-tune all layers with discriminative LR |
| Large dataset, abundant compute | Train from scratch (still use pretrained init) |

```python
# Strategy 1: Freeze backbone, train head only
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():   # or model.head, model.classifier
    param.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Strategy 2: Discriminative learning rates (different LR per layer group)
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])

# Learning rate schedule: warmup + cosine decay
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.05  # 5% warmup
)
```

---

## CV Evaluation Metrics

### Classification

```python
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# top-1 accuracy
top1 = accuracy_score(y_true, y_pred_classes)

# top-5 accuracy (multi-class problems with many classes)
top5 = top_k_accuracy_score(y_true, y_prob, k=5)
print(f"Top-1: {top1:.4f}, Top-5: {top5:.4f}")
```

### Detection: IoU, mAP

```python
import torchvision.ops as ops
import torch

def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    Returns IoU scalar.
    """
    b1 = torch.tensor(box1, dtype=torch.float).unsqueeze(0)
    b2 = torch.tensor(box2, dtype=torch.float).unsqueeze(0)
    return ops.box_iou(b1, b2).item()

# mAP with torchmetrics (recommended)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type='bbox')

# preds: list of dicts with keys 'boxes', 'scores', 'labels'
# targets: list of dicts with keys 'boxes', 'labels'
metric.update(preds, targets)
result = metric.compute()
print(f"mAP@0.5:     {result['map_50']:.4f}")
print(f"mAP@0.5:0.95:{result['map']:.4f}")
```

IoU threshold explanation:
- **IoU@0.5** (PASCAL VOC): a detection is correct if the box overlaps ground truth by ≥50%
- **mAP@0.5:0.95** (COCO): average mAP over thresholds 0.5, 0.55, …, 0.95 — stricter, preferred for modern benchmarks

### Segmentation: mIoU, Dice

```python
def compute_miou(pred_mask, true_mask, num_classes):
    """pred_mask, true_mask: integer class labels, shape [H, W]"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        if union == 0:
            continue  # class not present in either — skip
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0

def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask * true_mask).sum()
    return (2 * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)
```

### Generation / Synthesis: FID, SSIM, LPIPS

```python
# SSIM — structural similarity, higher is better (max 1.0)
from skimage.metrics import structural_similarity as ssim
import cv2

img1 = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('reconstructed.png', cv2.IMREAD_GRAYSCALE)
score = ssim(img1, img2, data_range=255)
print(f"SSIM: {score:.4f}")

# FID — Frechet Inception Distance (lower is better)
# Install: pip install pytorch-fid
# Usage from command line:
#   python -m pytorch_fid path/to/real_images path/to/generated_images

# LPIPS — perceptual similarity (lower is better)
# Install: pip install lpips
import lpips
import torch

loss_fn = lpips.LPIPS(net='alex')  # or 'vgg'
img1_t = torch.from_numpy(img1_rgb).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
img2_t = torch.from_numpy(img2_rgb).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
d = loss_fn(img1_t, img2_t)
print(f"LPIPS: {d.item():.4f}")
```

Metric summary:
| Metric | Task | Better when |
|--------|------|-------------|
| Top-1 / Top-5 accuracy | Classification | Higher |
| mAP@0.5, mAP@0.5:0.95 | Detection | Higher |
| mIoU | Segmentation | Higher |
| Dice | Segmentation | Higher |
| FID | Generation | Lower |
| SSIM | Reconstruction | Higher |
| LPIPS | Perceptual quality | Lower |

---

## Best Practices

1. **Profile the dataset before writing training code** — size variance and class imbalance dictate key pipeline choices
2. **OpenCV reads BGR** — always convert to RGB before any DL library
3. **Use albumentations for detection/segmentation** — spatial transforms are applied consistently to boxes and masks
4. **Validate augmentation visually** — render 10 augmented samples before training to catch errors
5. **Match normalization to pretraining** — ImageNet stats for ImageNet-pretrained, dataset-specific otherwise
6. **Monitor training loss AND sample predictions** — loss curves alone miss silent failures (mode collapse, label errors)
7. **Use mAP@0.5:0.95 for detection benchmarks** — mAP@0.5 alone is too lenient for modern standards

---

## Common Pitfalls

- **Augmenting validation data**: Apply only resize + normalize to val/test; augmentations go on train only
- **Wrong normalization order**: Normalize AFTER ToTensor (images in [0,1]) not before
- **Not handling EXIF rotation**: PIL respects EXIF by default; OpenCV does not — check orientation
- **FID computed on too few images**: FID is unreliable below ~10k samples; use at least 10k real + 10k generated
- **Comparing mAP across IoU thresholds without clarifying**: Always state which threshold you report
