import torch
import open_clip
import numpy as np
from PIL import Image


import warnings
warnings.filterwarnings("ignore")


class ZeroShotVideoClassifier:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    def encode_frames(self, frames: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(f) for f in frames]).to(self.device)

        with torch.no_grad():
            frame_embeddings = self.model.encode_image(tensors)
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)

        return frame_embeddings.mean(dim=0)

    def encode_labels(self, labels: list[str], prompt_template: str = "a video of {}") -> torch.Tensor:
        texts = self.tokenizer([prompt_template.format(l) for l in labels]).to(self.device)

        with torch.no_grad():
            text_embeddings = self.model.encode_text(texts)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return text_embeddings

    def classify(self, frames: list[Image.Image], labels: list[str], top_k: int = 5) -> list[dict]:
        video_embedding = self.encode_frames(frames)
        text_embeddings = self.encode_labels(labels)

        similarities = (video_embedding @ text_embeddings.T).squeeze()
        probs = similarities.softmax(dim=-1).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]

        return [
            {"label": labels[i], "score": float(probs[i])}
            for i in top_indices
        ]


if __name__ == "__main__":
    from frame_extractor import extract_frames
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <video_path>")
        sys.exit(1)

    labels = [
        "playing basketball", "playing guitar", "cooking food",
        "swimming", "riding a bike", "doing archery", "weightlifting"
    ]

    classifier = ZeroShotVideoClassifier()
    frames = extract_frames(sys.argv[1], num_frames=8)
    results = classifier.classify(frames, labels, top_k=3)

    for r in results:
        print(f"{r['label']:<30} {r['score']:.4f}")
