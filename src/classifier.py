import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image


MODELS = {
    "CLIP ViT-B/32": "clip",
    "SigLIP 2 Base": "siglip2",
    "X-CLIP Base": "xclip",
}


class ZeroShotVideoClassifier:
    def __init__(self, model_key: str = "CLIP ViT-B/32", device: str = None):
        self.model_key = model_key
        self.backend = MODELS[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        if self.backend == "clip":
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model.to(self.device).eval()

        elif self.backend == "siglip2":
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
            self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            self.model.to(self.device)

        elif self.backend == "xclip":
            from transformers import XCLIPModel, XCLIPProcessor
            self.model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16").eval()
            self.processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16")
            self.model.to(self.device)

    def _encode_text_clip(self, labels: list[str]) -> torch.Tensor:
        texts = self.tokenizer(["a video of {}".format(l) for l in labels]).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_text(texts)
            return embs / embs.norm(dim=-1, keepdim=True)

    def _encode_frames_clip(self, frames: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_image(tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            return embs.mean(dim=0, keepdim=True)

    def _encode_text_siglip2(self, labels: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=["a video of {}".format(l) for l in labels],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embs = out.pooler_output if hasattr(out, "pooler_output") else out[0][:, 0, :]
            return embs / embs.norm(dim=-1, keepdim=True)

    def _encode_frames_siglip2(self, frames: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            embs = out.pooler_output if hasattr(out, "pooler_output") else out[0][:, 0, :]
            embs = embs / embs.norm(dim=-1, keepdim=True)
            video_emb = embs.mean(dim=0, keepdim=True)
            return video_emb / video_emb.norm(dim=-1, keepdim=True)

    def _encode_text_xclip(self, labels: list[str]) -> torch.Tensor:
        inputs = self.processor.tokenizer(
            ["a video of {}".format(l) for l in labels],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embs = out.pooler_output
            return embs / embs.norm(dim=-1, keepdim=True)

    def _encode_frames_xclip(self, frames: list[Image.Image]) -> torch.Tensor:
        frames_np = [np.array(f) for f in frames]
        inputs = self.processor.image_processor(images=frames_np, return_tensors="pt")
        with torch.no_grad():
            out = self.model.get_video_features(pixel_values=inputs["pixel_values"])
            embs = out.pooler_output
            return embs / embs.norm(dim=-1, keepdim=True)

    def classify(self, frames: list[Image.Image], labels: list[str], top_k: int = 5) -> list[dict]:
        if self.backend == "clip":
            text_embs = self._encode_text_clip(labels)
            video_emb = self._encode_frames_clip(frames)
            scores = (video_emb @ text_embs.T).squeeze().softmax(dim=-1).cpu().numpy()

        elif self.backend == "siglip2":
            text_embs = self._encode_text_siglip2(labels)
            video_emb = self._encode_frames_siglip2(frames)
            logit_scale = self.model.logit_scale.exp()
            logit_bias = self.model.logit_bias
            logits = video_emb @ text_embs.T * logit_scale + logit_bias
            scores = torch.sigmoid(logits).detach().squeeze().cpu().numpy()

        elif self.backend == "xclip":
            text_embs = self._encode_text_xclip(labels)
            video_emb = self._encode_frames_xclip(frames)
            scores = (video_emb @ text_embs.T).detach().squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"label": labels[i], "score": float(scores[i])} for i in top_indices]
