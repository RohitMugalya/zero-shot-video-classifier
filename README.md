---
title: Zero-Shot Video Classifier
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Zero-Shot Video Classifier

Classify any video using natural language — no task-specific training required.

## Models

| Model | Size | UCF-101 Top-1 | UCF-101 Top-5 | Type |
|---|---|---|---|---|
| CLIP ViT-B/32 | 338MB | 58.22% | 85.35% | Pure zero-shot |
| SigLIP 2 Base | 350MB | 70.79% | 93.27% | Pure zero-shot |
| X-CLIP Base | 780MB | 72.44% | 91.24% | Kinetics pretrained |

## How it works

1. Upload any video
2. Enter natural language labels (one per line)
3. Select a model
4. Click Run — the model ranks your labels by similarity to the video content

No fine-tuning, no training data needed. CLIP and SigLIP 2 encode frames independently
and average embeddings. X-CLIP uses cross-frame attention to model temporal dynamics.

## Benchmark

Evaluated on UCF-101 test split — 10 videos per class, 101 classes, 1010 videos total.

> **Note:** X-CLIP was pretrained on Kinetics-400 which has class overlap with UCF-101,
> so its benchmark score is not a clean zero-shot result unlike CLIP and SigLIP 2.

## Stack

`PyTorch` · `OpenCLIP` · `HuggingFace Transformers` · `Streamlit` · `OpenCV`

## Author

[RohitMugalya](https://huggingface.co/RohitMugalya) · [GitHub](https://github.com/RohitMugalya/zero-shot-video-classifier)