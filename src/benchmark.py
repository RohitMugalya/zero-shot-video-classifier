import warnings
warnings.filterwarnings("ignore")

import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import yaml
import json
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_ucf101_classes(classlist_path: str) -> dict[int, str]:
    classes = {}
    with open(classlist_path) as f:
        for line in f:
            idx, name = line.strip().split()
            classes[int(idx)] = name.replace("_", " ").lower()
    return classes


def load_test_videos(testlist_path: str, ucf101_root: str, max_per_class: int = 10) -> list[dict]:
    root = Path(ucf101_root)
    videos = []
    class_counts = {}

    with open(testlist_path) as f:
        for line in f:
            rel_path = line.strip()
            class_name = rel_path.split("/")[0].replace("_", " ").lower()

            if class_counts.get(class_name, 0) >= max_per_class:
                continue

            full_path = root / rel_path
            if full_path.exists():
                videos.append({"path": str(full_path), "label": class_name})
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return videos


def extract_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def run_benchmark(config: dict, device: str) -> pd.DataFrame:
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model"]["name"], pretrained=config["model"]["pretrained"]
    )
    tokenizer = open_clip.get_tokenizer(config["model"]["name"])
    model.to(device).eval()

    classes = load_ucf101_classes(config["dataset"]["classlist_path"])
    label_list = [classes[k] for k in sorted(classes)]
    template = config["model"]["prompt_template"]

    texts = tokenizer([template.format(l) for l in label_list]).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    videos = load_test_videos(
        config["dataset"]["testlist_path"],
        config["dataset"]["ucf101_root"],
        config["dataset"]["max_videos_per_class"]
    )

    results = []

    for video in tqdm(videos, desc="Evaluating"):
        frames = extract_frames(video["path"], config["extraction"]["num_frames"])
        if not frames:
            continue

        tensors = torch.stack([preprocess(f) for f in frames]).to(device)
        with torch.no_grad():
            frame_embs = model.encode_image(tensors)
            frame_embs = frame_embs / frame_embs.norm(dim=-1, keepdim=True)
            video_emb = frame_embs.mean(dim=0)

        sims = (video_emb @ text_embeddings.T).cpu().numpy()
        top5_indices = np.argsort(sims)[::-1][:5]
        top5_labels = [label_list[i] for i in top5_indices]

        results.append({
            "video": Path(video["path"]).name,
            "true_label": video["label"],
            "top1_pred": top5_labels[0],
            "top5_preds": top5_labels,
            "top1_correct": video["label"] == top5_labels[0],
            "top5_correct": video["label"] in top5_labels,
        })

    return pd.DataFrame(results)


def compute_metrics(df: pd.DataFrame) -> dict:
    top1 = df["top1_correct"].mean() * 100
    top5 = df["top5_correct"].mean() * 100

    per_class = df.groupby("true_label")["top1_correct"].mean() * 100
    best_classes = per_class.nlargest(5).to_dict()
    worst_classes = per_class.nsmallest(5).to_dict()

    return {
        "top1_accuracy": round(top1, 2),
        "top5_accuracy": round(top5, 2),
        "total_videos": len(df),
        "best_classes": best_classes,
        "worst_classes": worst_classes,
    }


if __name__ == "__main__":
    import os
    config = load_config()
    os.makedirs(config["evaluation"]["output_path"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    df = run_benchmark(config, device)
    metrics = compute_metrics(df)

    df.to_csv(f"{config['evaluation']['output_path']}/results.csv", index=False)
    with open(f"{config['evaluation']['output_path']}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTop-1 Accuracy : {metrics['top1_accuracy']}%")
    print(f"Top-5 Accuracy : {metrics['top5_accuracy']}%")
    print(f"Total Videos   : {metrics['total_videos']}")
