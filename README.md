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

Zero-shot video classification benchmarking CLIP ViT-B/32, SigLIP 2, and X-CLIP on UCF-101 — no task-specific training required.

![Project Thumbnail](assets/images/thumbnail.png)

## Live Demo

👉 **[Try it on HuggingFace Spaces](https://huggingface.co/spaces/RohitMugalya/zero-shot-video-classifier)**

> **Note:** The Space runs on a free CPU tier and may go inactive after periods of no use.
> If you see a sleeping screen, click **Restart** and wait ~2 minutes for it to wake up.
> The first inference after a cold start takes an additional 1–3 minutes as the selected
> model downloads and loads into memory. Subsequent classifications in the same session are fast.

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

No fine-tuning, no training data needed. CLIP and SigLIP 2 encode frames independently and average embeddings. X-CLIP uses cross-frame attention to model temporal dynamics.

## Benchmark

Evaluated on UCF-101 test split — 10 videos per class, 101 classes, 1010 videos total.

> **Note:** X-CLIP was pretrained on Kinetics-400 which has class overlap with UCF-101,
> so its benchmark score is not a clean zero-shot result unlike CLIP and SigLIP 2.

## Stack

`PyTorch` · `OpenCLIP` · `HuggingFace Transformers` · `Streamlit` · `OpenCV`

## Author

[RohitMugalya](https://huggingface.co/RohitMugalya) · [GitHub](https://github.com/RohitMugalya/zero-shot-video-classifier)
