import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import torch
import tempfile
import os
import time

from classifier import ZeroShotVideoClassifier
from frame_extractor import extract_frames

st.set_page_config(
    page_title="Zero-Shot Video Classifier",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #a0a0c0;
    margin-bottom: 2rem;
}

.metric-card {
    background: #13131f;
    border: 1px solid #1e1e32;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7c6af7;
}

.metric-label {
    font-size: 0.8rem;
    color: #6b6b88;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

.result-bar-wrap {
    background: #13131f;
    border: 1px solid #1e1e32;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.result-label {
    font-size: 0.95rem;
    font-weight: 500;
    color: #e8e8f0;
    margin-bottom: 0.4rem;
    text-transform: capitalize;
}

.result-bar-bg {
    background: #1e1e32;
    border-radius: 4px;
    height: 8px;
    width: 100%;
}

.result-bar-fill {
    background: linear-gradient(90deg, #7c6af7, #a78bfa);
    border-radius: 4px;
    height: 8px;
}

.result-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #7c6af7;
    margin-top: 0.3rem;
}

.badge {
    display: inline-block;
    background: #1e1e32;
    border: 1px solid #2e2e48;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #9090b8;
    margin-right: 0.4rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #7878a8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

.frame-caption {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7878a8;
    text-align: center;
    margin-top: 0.3rem;
}

div[data-testid="stFileUploader"] {
    background: #13131f;
    border: 1px dashed #2e2e48;
    border-radius: 12px;
    padding: 1rem;
}

div[data-testid="stTextInput"] input {
    background: #13131f;
    border: 1px solid #2e2e48;
    border-radius: 8px;
    color: #e8e8f0;
    font-family: 'DM Sans', sans-serif;
}

.stButton > button {
    background: #7c6af7;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #6b58e8;
}

hr {
    border-color: #1e1e32;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    return ZeroShotVideoClassifier(model_name="ViT-B-32", pretrained="openai")


st.markdown('<div class="hero-title">Zero-Shot<br>Video Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Classify any video with natural language — no training required</div>', unsafe_allow_html=True)

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.markdown('<span class="badge">CLIP ViT-B/32</span>', unsafe_allow_html=True)
with col_meta2:
    st.markdown('<span class="badge">UCF-101: 58.2% Top-1</span>', unsafe_allow_html=True)
with col_meta3:
    st.markdown('<span class="badge">Zero-Shot</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Labels — comma separated</div>', unsafe_allow_html=True)

    default_labels = "playing basketball, swimming, cooking food, riding a bike, doing archery, playing guitar, weightlifting, dancing"
    labels_input = st.text_input(
        "Labels",
        value=default_labels,
        label_visibility="collapsed",
        placeholder="e.g. playing basketball, swimming, cooking..."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Frames to sample</div>', unsafe_allow_html=True)
    num_frames = st.slider("Frames", min_value=4, max_value=16, value=8, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Classify Video")

with right_col:
    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        labels = [l.strip() for l in labels_input.split(",") if l.strip()]

        if len(labels) < 2:
            st.error("Please enter at least 2 labels.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Loading model..."):
                classifier = load_model()

            with st.spinner("Extracting frames..."):
                frames = extract_frames(tmp_path, num_frames=num_frames)

            if not frames:
                st.error("Could not extract frames from this video.")
            else:
                with st.spinner("Classifying..."):
                    start = time.time()
                    results = classifier.classify(frames, labels, top_k=len(labels))
                    elapsed = time.time() - start

                os.unlink(tmp_path)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(frames)}</div><div class="metric-label">Frames Used</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.1f}s</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)
                with m3:
                    top_score = results[0]["score"] * 100
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{top_score:.1f}%</div><div class="metric-label">Top-1 Confidence</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                for i, r in enumerate(results):
                    score_pct = r["score"] * 100
                    bar_width = int(score_pct)
                    rank_color = "#7c6af7" if i == 0 else "#3a3a58"
                    st.markdown(f"""
                    <div class="result-bar-wrap">
                        <div class="result-label">{"🥇 " if i == 0 else f"#{i+1}  "}{r["label"]}</div>
                        <div class="result-bar-bg">
                            <div class="result-bar-fill" style="width:{bar_width}%; background: {'linear-gradient(90deg,#7c6af7,#a78bfa)' if i == 0 else '#2e2e48'};"></div>
                        </div>
                        <div class="result-score">{score_pct:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Sampled Frames</div>', unsafe_allow_html=True)
                frame_cols = st.columns(min(len(frames), 8))
                for i, (col, frame) in enumerate(zip(frame_cols, frames)):
                    with col:
                        st.image(frame, use_container_width=True)
                        st.markdown(f'<div class="frame-caption">f{i+1}</div>', unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="background:#13131f; border:1px solid #1e1e32; border-radius:12px; padding:3rem 2rem; text-align:center; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🎬</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#6868a0;">
                Upload a video and enter labels<br>to see predictions here
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.7rem; color:#3a3a58;">
    CLIP ViT-B/32 &nbsp;·&nbsp; UCF-101 Benchmark: 58.22% Top-1 &nbsp;·&nbsp; 85.35% Top-5 &nbsp;·&nbsp; Zero-Shot · No Training Required
</div>
""", unsafe_allow_html=True)
