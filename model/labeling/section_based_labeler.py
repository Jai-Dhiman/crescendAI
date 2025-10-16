#!/usr/bin/env python3
"""
Section-Based Labeler for CrescendAI Evaluator

This labeler organizes segments by piece and section (bars) rather than sequentially,
allowing you to label the same musical passage across all performers before moving
to the next section. This enables more consistent comparative labeling.

Usage:
    streamlit run labeling/section_based_labeler.py
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

import streamlit as st
import soundfile as sf
import numpy as np

# Optional: use torchaudio if you prefer its loaders (commented here)
try:
    import torchaudio  # noqa: F401
except Exception:
    torchaudio = None  # Not required for basic playbook


@dataclass
class SegmentGroup:
    """Group of segments representing the same section across different performers"""
    piece: str
    bars: Tuple[int, int]
    segments: List[Dict]  # List of segments from different performers
    
    def __str__(self):
        return f"{self.piece} | bars {self.bars[0]}-{self.bars[1]} | {len(self.segments)} performers"


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


def group_segments_by_section(segments: List[Dict]) -> List[SegmentGroup]:
    """
    Group segments by piece and bar range for section-based labeling.
    
    Returns segments organized so you can label the same musical section
    across all performers before moving to the next section.
    """
    # Group by (piece, bars)
    groups = defaultdict(list)
    
    for segment in segments:
        piece = segment["provenance"]["piece"]
        bars = tuple(segment["bars"])
        groups[(piece, bars)].append(segment)
    
    # Convert to SegmentGroup objects and sort
    segment_groups = []
    for (piece, bars), segments_list in groups.items():
        # Sort performers within each section by year for consistent ordering
        segments_list.sort(key=lambda x: x["provenance"]["year"])
        segment_groups.append(SegmentGroup(piece, bars, segments_list))
    
    # Sort groups by piece name, then by starting bar
    segment_groups.sort(key=lambda g: (g.piece, g.bars[0]))
    
    return segment_groups


def resolve_file_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri.replace("file://", "")
    return uri


def safe_slice_audio(
    path: Path, t0: float, t1: float, target_sr: Optional[int] = None
) -> Optional[bytes]:
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


def merge_labels(
    orig: Dict, new_labels: Dict[str, float], new_mask: Dict[str, int]
) -> Dict:
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


def render_segment_card(segment: Dict, dims: List[str], anchors: Dict, key_prefix: str):
    """Render an individual segment with its controls in a card format"""
    year = segment["provenance"]["year"]
    segment_id = segment.get("segment_id", "")
    
    with st.container():
        st.markdown(f"**Performer: {year}**")
        
        # Audio playback
        uri = segment.get("audio_uri", "")
        t0 = float(segment.get("t0", 0.0))
        t1 = float(segment.get("t1", max(0.1, t0 + 3.0)))
        audio_path = Path(resolve_file_uri(uri))
        
        wav_bytes = safe_slice_audio(audio_path, t0, t1, target_sr=22050)
        if wav_bytes is not None:
            st.audio(wav_bytes, format="audio/wav")
        else:
            st.warning(f"Unable to play audio for performer {year}")
        
        # Dimension sliders
        new_labels: Dict[str, float] = {}
        new_mask: Dict[str, int] = {}
        
        for d in dims:
            cur_val = float((segment.get("labels") or {}).get(d, 0.5))
            cur_mask = bool((segment.get("label_mask") or {}).get(d, 0))
            
            col1, col2 = st.columns([4, 1])
            with col1:
                v = st.slider(
                    f"{d}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cur_val),
                    step=0.01,
                    key=f"slider_{key_prefix}_{segment_id}_{d}",
                )
            with col2:
                m = st.checkbox(
                    "Labeled",
                    value=cur_mask,
                    key=f"mask_{key_prefix}_{segment_id}_{d}",
                )
            new_labels[d] = float(v)
            new_mask[d] = 1 if m else 0
        
        return new_labels, new_mask


# ---- Streamlit App ----
st.set_page_config(page_title="CrescendAI Section-Based Labeler", layout="wide")
st.title("CrescendAI: Section-Based Evaluator Labeler")

st.markdown("""
**Section-based labeling workflow:** Label the same musical section across all performers 
before moving to the next section. This enables more consistent comparative evaluation.
""")

# Sidebar: manifest selection
with st.sidebar:
    st.header("Data")
    default_manifests = []
    for p in ["data/manifests/segments_unlabeled.jsonl", "data/splits/train.jsonl", "data/splits/valid.jsonl"]:
        if Path(p).exists():
            default_manifests.append(p)
    manifest_path_str = st.selectbox(
        "Manifest JSONL",
        options=(
            (default_manifests + ["<custom path>"])
            if default_manifests
            else ["<custom path>"]
        ),
        index=0,
    )
    if manifest_path_str == "<custom path>":
        manifest_path_str = st.text_input("Enter path to manifest JSONL", value="")
    manifest_path = Path(manifest_path_str) if manifest_path_str else None

    st.header("Output")
    out_path_str = st.text_input(
        "Save labeled rows to",
        value="data/manifests/segments_labeled_sectional.jsonl",
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

# Load manifest and group by sections
try:
    segments = load_jsonl(manifest_path)
except Exception as e:
    st.error(str(e))
    st.stop()

if not segments:
    st.warning("Manifest is empty.")
    st.stop()

# Group segments by section
section_groups = group_segments_by_section(segments)

if not section_groups:
    st.warning("No section groups found.")
    st.stop()

# Session state for section index
if "section_idx" not in st.session_state:
    st.session_state.section_idx = 0

# Section navigation controls
st.markdown("### Navigation")
cols = st.columns([1, 1, 4, 1, 1])
with cols[0]:
    if st.button("Previous Section", use_container_width=True):
        st.session_state.section_idx = max(0, st.session_state.section_idx - 1)
with cols[1]:
    if st.button("Next Section", use_container_width=True):
        st.session_state.section_idx = min(len(section_groups) - 1, st.session_state.section_idx + 1)

with cols[3]:
    # Jump to specific section
    section_names = [f"Section {i+1}: {str(group)}" for i, group in enumerate(section_groups)]
    selected_section = st.selectbox(
        "Jump to:",
        options=range(len(section_groups)),
        format_func=lambda i: section_names[i],
        index=st.session_state.section_idx,
        key="section_selector"
    )
    if selected_section != st.session_state.section_idx:
        st.session_state.section_idx = selected_section

with cols[4]:
    st.metric("Progress", f"{st.session_state.section_idx + 1}/{len(section_groups)}")

# Current section group
current_group = section_groups[st.session_state.section_idx]
st.markdown(f"## {current_group}")

# Get dimensions from the first segment (should be consistent)
dims = list(current_group.segments[0].get("dims", []))
if not dims:
    st.warning("This section has no dims field; add dims to your manifest for a better UI.")

# Anchors panel (if available)
if anchors and dims:
    with st.expander("Reference Anchors", expanded=False):
        sel_dim = st.selectbox("Dimension", options=dims, key="anchor_dim_selector")
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

# Render all performers for this section side by side
st.markdown("### Comparative Labeling")
st.markdown("**Instructions:** Listen to each performer's rendition of this section and label consistently across all performers.")

# Store all labels for saving
all_labels = {}
all_masks = {}

# Create columns for side-by-side comparison
if len(current_group.segments) <= 3:
    # Show all performers in one row if 3 or fewer
    performer_cols = st.columns(len(current_group.segments))
    for i, segment in enumerate(current_group.segments):
        with performer_cols[i]:
            labels, mask = render_segment_card(
                segment, dims, anchors, f"section_{st.session_state.section_idx}_performer_{i}"
            )
            all_labels[segment["segment_id"]] = labels
            all_masks[segment["segment_id"]] = mask
else:
    # Show in rows of 2 if more than 3 performers
    for i in range(0, len(current_group.segments), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(current_group.segments):
                segment = current_group.segments[i + j]
                with col:
                    labels, mask = render_segment_card(
                        segment, dims, anchors, f"section_{st.session_state.section_idx}_performer_{i+j}"
                    )
                    all_labels[segment["segment_id"]] = labels
                    all_masks[segment["segment_id"]] = mask

# Save controls
st.divider()
col_save, col_save_next, col_skip = st.columns([1, 1, 1])

with col_save:
    if st.button("Save All Labels", use_container_width=True):
        try:
            saved_count = 0
            for segment in current_group.segments:
                segment_id = segment["segment_id"]
                if segment_id in all_labels:
                    updated = merge_labels(segment, all_labels[segment_id], all_masks[segment_id])
                    append_jsonl(out_path, updated)
                    saved_count += 1
            st.success(f"Saved {saved_count} segments to {out_path}")
        except Exception as e:
            st.error(f"Failed to save: {e}")

with col_save_next:
    if st.button("Save & Next Section", use_container_width=True):
        try:
            saved_count = 0
            for segment in current_group.segments:
                segment_id = segment["segment_id"]
                if segment_id in all_labels:
                    updated = merge_labels(segment, all_labels[segment_id], all_masks[segment_id])
                    append_jsonl(out_path, updated)
                    saved_count += 1
            st.success(f"Saved {saved_count} segments")
            st.session_state.section_idx = min(len(section_groups) - 1, st.session_state.section_idx + 1)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to save: {e}")

with col_skip:
    if st.button("Skip Section", use_container_width=True):
        st.session_state.section_idx = min(len(section_groups) - 1, st.session_state.section_idx + 1)
        st.experimental_rerun()

# Progress summary
st.divider()
with st.expander("Session Summary", expanded=False):
    st.write(f"**Total sections:** {len(section_groups)}")
    st.write(f"**Current section:** {st.session_state.section_idx + 1}")
    st.write(f"**Sections remaining:** {len(section_groups) - st.session_state.section_idx - 1}")
    st.write(f"**Current piece:** {current_group.piece}")
    st.write(f"**Current bars:** {current_group.bars[0]}-{current_group.bars[1]}")
    st.write(f"**Performers in this section:** {len(current_group.segments)}")
    
    # Show all pieces and their section counts
    piece_counts = defaultdict(int)
    for group in section_groups:
        piece_counts[group.piece] += 1
    
    st.write("**Pieces in dataset:**")
    for piece, count in piece_counts.items():
        st.write(f"- {piece}: {count} sections")