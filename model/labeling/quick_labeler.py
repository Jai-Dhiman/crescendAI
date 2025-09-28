#!/usr/bin/env python3
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import soundfile as sf
import numpy as np

# Optional: use torchaudio if you prefer its loaders (commented here)
try:
    import torchaudio  # noqa: F401
except Exception:
    torchaudio = None  # Not required for basic playback


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln} in {path}: {e}")
    return rows


def resolve_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri.replace("file://", "")
    return uri


def safe_slice_audio(path: Path, t0: float, t1: float, target_sr: Optional[int] = None) -> Optional[bytes]:
    """Load and slice audio to [t0, t1], return WAV bytes suitable for st.audio.
    Explicit exceptions where appropriate per user preference; otherwise return None on recoverable cases.
    """
    try:
        # soundfile reads and returns (numpy array, sr)
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data is None or len(data) == 0:
            st.error(f"Empty audio: {path}")
            return None
        if data.ndim > 1:
            # Mix to mono
            data = data.mean(axis=1)
        if target_sr is not None and sr != target_sr:
            try:
                import resampy
            except Exception:
                st.warning("resampy not installed; keeping original sample rate.")
                target_sr = sr
            else:
                data = resampy.resample(data, sr, target_sr)
                sr = target_sr
        s0 = max(0, int(t0 * sr))
        s1 = min(len(data), int(t1 * sr))
        if s1 <= s0:
            st.error(f"Invalid time bounds for {path}: t0={t0}, t1={t1}, sr={sr}")
            return None
        clip = data[s0:s1]
        # Write to WAV in-memory
        buf = io.BytesIO()
        sf.write(file=buf, data=clip, samplerate=sr, format="WAV")
        buf.seek(0)
        return buf.read()
    except FileNotFoundError:
        st.error(f"Audio file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Failed to load audio {path}: {type(e).__name__}: {e}")
        return None


def load_anchors(anchors_path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    if not anchors_path.exists():
        return {}
    try:
        return json.loads(anchors_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read anchors at {anchors_path}: {e}")
        return {}


def merge_labels(orig: Dict, new_labels: Dict[str, float], new_mask: Dict[str, int]) -> Dict:
    eg = dict(orig)
    labels = dict(eg.get("labels") or {})
    labels.update(new_labels)
    eg["labels"] = labels
    mask = dict(eg.get("label_mask") or {})
    mask.update(new_mask)
    eg["label_mask"] = mask
    eg["source"] = "human"
    return eg


def append_jsonl(path: Path, obj: Dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Failed to append to {path}: {e}")


# ---- Streamlit App ----
st.set_page_config(page_title="CrescendAI Quick Labeler", layout="wide")
st.title("CrescendAI: Evaluator Quick Labeler")

# Sidebar: manifest selection
with st.sidebar:
    st.header("Data")
    default_manifests = []
    for p in ["data/splits/train.jsonl", "data/splits/valid.jsonl"]:
        if Path(p).exists():
            default_manifests.append(p)
    manifest_path_str = st.selectbox(
        "Manifest JSONL",
        options=(default_manifests + ["<custom path>"])
        if default_manifests
        else ["<custom path>"],
        index=0,
    )
    if manifest_path_str == "<custom path>":
        manifest_path_str = st.text_input("Enter path to manifest JSONL", value="")
    manifest_path = Path(manifest_path_str) if manifest_path_str else None

    st.header("Output")
    out_path_str = st.text_input(
        "Save labeled rows to",
        value="data/manifests/segments_labeled.jsonl",
    )
    out_path = Path(out_path_str) if out_path_str else None

    st.header("Anchors")
    anchors_path_str = st.text_input(
        "Anchors JSON (optional)", value="data/anchors/anchors.json"
    )
    anchors = load_anchors(Path(anchors_path_str))

if not manifest_path or not out_path:
    st.info("Select a manifest and output path from the sidebar to begin.")
    st.stop()

# Load manifest
try:
    rows = load_jsonl(manifest_path)
except Exception as e:
    st.error(str(e))
    st.stop()

if not rows:
    st.warning("Manifest is empty.")
    st.stop()

# Session state for index
if "idx" not in st.session_state:
    st.session_state.idx = 0

# Navigation controls
cols = st.columns([1, 1, 6, 1])
with cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
with cols[3]:
    if st.button("Next", use_container_width=True):
        st.session_state.idx = min(len(rows) - 1, st.session_state.idx + 1)

# Current example
i = st.session_state.idx
example = rows[i]
st.write(f"Row {i+1}/{len(rows)} — segment_id: {example.get('segment_id','<none>')}")

# Audio panel
uri = example.get("audio_uri", "")
t0 = float(example.get("t0", 0.0))
t1 = float(example.get("t1", max(0.1, t0 + 3.0)))
audio_path = Path(resolve_file_uri(uri))

with st.expander("Audio", expanded=True):
    wav_bytes = safe_slice_audio(audio_path, t0, t1, target_sr=22050)
    if wav_bytes is not None:
        st.audio(wav_bytes, format="audio/wav")
    else:
        st.warning("Unable to play audio for this segment.")

# Dimensions and labeling UI
dims: List[str] = list(example.get("dims") or [])
if not dims:
    st.warning("This segment has no dims field; add dims to your manifest for a better UI.")

# Anchors panel per-dimension (if available)
if anchors:
    with st.expander("Anchors", expanded=False):
        sel_dim = st.selectbox("Dimension", options=dims or list(anchors.keys()))
        if sel_dim and sel_dim in anchors:
            ac = anchors[sel_dim]
            ac_cols = st.columns(3)
            for j, bucket in enumerate(["low", "mid", "high"]):
                with ac_cols[j]:
                    st.caption(f"{sel_dim} — {bucket}")
                    clip_uri = (ac.get(bucket) or {}).get("clip_uri") or ""
                    desc = (ac.get(bucket) or {}).get("desc") or ""
                    st.write(desc)
                    if clip_uri:
                        clip_path = Path(resolve_file_uri(clip_uri))
                        if clip_path.exists():
                            try:
                                b = safe_slice_audio(clip_path, 0.0, float("inf"))
                                if b is not None:
                                    st.audio(b, format="audio/wav")
                            except Exception as e:
                                st.warning(f"Failed to load anchor {bucket}: {e}")
                        else:
                            st.warning(f"Anchor clip not found: {clip_path}")
        else:
            st.info("No anchors for the selected dimension.")

# Sliders and label_mask
st.subheader("Labels")
new_labels: Dict[str, float] = {}
new_mask: Dict[str, int] = {}

for d in dims:
    cur_val = float((example.get("labels") or {}).get(d, 0.5))
    col1, col2 = st.columns([4, 1])
    with col1:
        v = st.slider(
            f"{d}", min_value=0.0, max_value=1.0, value=float(cur_val), step=0.01
        )
    with col2:
        m = st.checkbox("Mark labeled", value=bool((example.get("label_mask") or {}).get(d, 0)))
    new_labels[d] = float(v)
    new_mask[d] = 1 if m else 0

# Save controls
st.divider()
col_save, col_skip = st.columns([1, 1])
with col_save:
    if st.button("Save label(s)"):
        try:
            updated = merge_labels(example, new_labels, new_mask)
            append_jsonl(out_path, updated)
            st.success(f"Saved to {out_path}")
        except Exception as e:
            st.error(f"Failed to save: {e}")
with col_skip:
    if st.button("Next segment"):
        st.session_state.idx = min(len(rows) - 1, st.session_state.idx + 1)
        st.experimental_rerun()
