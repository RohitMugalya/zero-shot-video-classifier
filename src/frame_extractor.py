import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def extract_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def extract_frames_from_folder(folder_path: str, num_frames: int = 8) -> dict[str, list[Image.Image]]:
    folder = Path(folder_path)
    video_extensions = {".avi", ".mp4", ".mov", ".mkv"}

    results = {}
    for video_file in folder.rglob("*"):
        if video_file.suffix.lower() in video_extensions:
            try:
                results[str(video_file)] = extract_frames(str(video_file), num_frames)
            except ValueError as e:
                print(f"Skipping {video_file.name}: {e}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path> [num_frames]")
        sys.exit(1)

    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    frames = extract_frames(path, n)
    print(f"Extracted {len(frames)} frames from {path}")
    print(f"Frame size: {frames[0].size}")
