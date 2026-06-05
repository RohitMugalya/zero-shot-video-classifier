import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import tempfile
import os
import time

from classifier import ZeroShotVideoClassifier, MODELS
from frame_extractor import extract_frames

st.set_page_config(
    page_title="Zero-Shot Video Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def load_classifier(model_key: str) -> ZeroShotVideoClassifier:
    return ZeroShotVideoClassifier(model_key=model_key)


st.title("🎬 Zero-Shot Video Classifier")
st.caption("Classify any video using natural language — no task-specific training required")

st.audio("assets/audios/instructions.mp3", format="audio/mp3")
st.divider()

tab1, tab2 = st.tabs(["Classification", "Model Information"])

with tab1:
    st.subheader("Input & configuration")
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        video_source = st.radio("Select Video Source:", ["Use Sample Video", "Upload Video"], horizontal=True)
        
        video_path = None
        tmp_path_to_delete = None

        if video_source == "Upload Video":
            uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path_to_delete = tmp.name
                    video_path = tmp.name
                st.video(video_path)
            else:
                st.info("Please upload a video.")
        else:
            sample_path = "assets/videos/sample_video.mp4"
            if os.path.exists(sample_path):
                video_path = sample_path
                st.video(video_path)
            else:
                st.warning(f"Sample video not found at {sample_path}")

    with right_col:
        labels_input = st.text_area(
            "Labels (one per line)",
            value="playing basketball\nswimming\ncooking food\nriding a bike\ndoing archery\nplaying guitar\nweightlifting\ndancing",
            height=200
        )
        
        can_run = video_path is not None
        run = st.button("▶  Run Classification", type="primary", disabled=not can_run, width="stretch")

        model_key = st.selectbox("Model", options=list(MODELS.keys()), index=0)
        model_info = {
            "CLIP ViT-B/32": "338MB · UCF-101 Top-1: 58.22% · Pure zero-shot",
            "SigLIP 2 Base": "350MB · UCF-101 Top-1: 70.79% · Pure zero-shot",
            "X-CLIP Base":   "780MB · UCF-101 Top-1: 72.44% · Kinetics pretrained",
        }
        st.caption(model_info[model_key])
        num_frames = st.slider("Frames to sample", min_value=4, max_value=16, value=8)

    if run and video_path:
        labels = [l.strip() for l in labels_input.splitlines() if l.strip()]

        if len(labels) < 2:
            st.error("Enter at least 2 labels.")
        else:
            with st.spinner(f"Loading {model_key}..."):
                classifier = load_classifier(model_key)

            with st.spinner("Extracting frames..."):
                frames = extract_frames(video_path, num_frames=num_frames)

            if not frames:
                st.error("Could not read frames from this video.")
            else:
                with st.spinner("Classifying..."):
                    t0 = time.time()
                    predictions = classifier.classify(frames, labels, top_k=len(labels))
                    elapsed = time.time() - t0

                st.divider()

                m1, m2, m3 = st.columns(3)
                m1.metric("Top Prediction", predictions[0]["label"].title())
                m2.metric("Confidence", f"{predictions[0]['score']*100:.1f}%")
                m3.metric("Inference Time", f"{elapsed:.1f}s")

                st.divider()
                st.subheader("Predictions")

                for i, r in enumerate(predictions):
                    rank = "🥇" if i == 0 else f"#{i+1}"
                    st.progress(
                        min(r["score"] * 2, 1.0),
                        text=f"{rank}  {r['label'].capitalize()}  —  {r['score']*100:.2f}%"
                    )

                st.divider()
                st.subheader("Sampled Frames")
                cols = st.columns(len(frames))
                for col, frame in zip(cols, frames):
                    with col:
                        st.image(frame, width="stretch")

        if tmp_path_to_delete and os.path.exists(tmp_path_to_delete):
            try:
                os.unlink(tmp_path_to_delete)
            except PermissionError:
                pass

with tab2:
    st.subheader("Benchmark — UCF-101")
    st.caption("10 videos per class · 101 classes · 1010 videos total")

    st.dataframe(
        data={
            "Model":  ["CLIP ViT-B/32", "SigLIP 2 Base", "X-CLIP Base"],
            "Size":   ["338MB", "350MB", "780MB"],
            "Top-1":  ["58.22%", "70.79%", "72.44%"],
            "Top-5":  ["85.35%", "93.27%", "91.24%"],
            "Type":   ["Pure zero-shot", "Pure zero-shot", "Kinetics pretrained"],
        },
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Model Notes")

    with st.expander("CLIP ViT-B/32", expanded=True):
        st.write("Contrastive softmax loss. Frames encoded independently then averaged into a single video embedding. Genuine zero-shot — no video-specific pretraining.")

    with st.expander("SigLIP 2 Base", expanded=True):
        st.write("Sigmoid loss objective instead of softmax. Scores per label are independent, making confidence values more calibrated. Best efficiency-to-accuracy ratio of the three.")

    with st.expander("X-CLIP Base", expanded=True):
        st.write("Video-native architecture with cross-frame attention — frames attend to each other before producing the final embedding. Pretrained on Kinetics-400 which has class overlap with UCF-101, so its score is not a clean zero-shot result.")
        st.warning("Kinetics-400 / UCF-101 class overlap — results may be inflated.", icon="⚠️")
