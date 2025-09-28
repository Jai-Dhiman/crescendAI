import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

# Master dimension list (keep in sync with training config)
ALL_DIMS: List[str] = [
    "timing_stability", "tempo_control", "rhythmic_accuracy",
    "articulation_length", "articulation_hardness",
    "pedal_density", "pedal_clarity",
    "dynamic_range", "dynamic_control",
    "balance_melody_vs_accomp",
    "phrasing_continuity", "expressiveness_intensity", "energy_level",
    "timbre_brightness", "timbre_richness", "timbre_color_variety",
]
DIM_TO_IDX = {d: i for i, d in enumerate(ALL_DIMS)}

DATASETS = ["MAESTRO", "ASAP", "MAPS", "CCMusic", "MusicNet", "YouTubeCurated"]
DS_TO_ID = {d: i for i, d in enumerate(DATASETS)}


class SegmentDataset(Dataset):
    """Loads audio segments and labels from a JSONL manifest.

    Expected fields per record:
    - segment_id, dataset, audio_uri, sr, t0, t1
    - dims: list[str] of candidate dims for this segment
    - labels: dict[dim -> value in 0..1] (optional)
    - label_mask: dict[dim -> 0/1] where 1 indicates a human label exists (optional)
    - pseudo_labels: dict[dim -> value] (optional)
    - pseudo_conf: dict[dim -> 0..1 confidence/weight] (optional)
    - distill_uri: optional path to a .npy vector (e.g., MidiBERT embedding, dim=256 suggested)
    """

    def __init__(
        self,
        manifest_path: str,
        sr: int = 22050,
        n_fft: int = 2048,
        hop: int = 512,
        n_mels: int = 128,
        seg_frames: int = 128,
        expected_distill_dim: int = 256,
    ):
        self.items: List[Dict[str, Any]] = [json.loads(l) for l in open(manifest_path)]
        self.sr = sr
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.seg_frames = seg_frames
        self.expected_distill_dim = expected_distill_dim

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _resolve_uri(uri: str) -> str:
        return uri.replace("file://", "")

    def _load_clip(self, uri: str, t0: float, t1: float, target_sr: int) -> torch.Tensor:
        wav, sr = torchaudio.load(self._resolve_uri(uri))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        s0, s1 = int(t0 * target_sr), int(t1 * target_sr)
        wav = wav[:, s0:s1]
        if wav.numel() == 0:
            # Guard: if times are out of range, pad silence
            wav = torch.zeros(1, max(1, s1 - s0), dtype=torch.float32)
        return wav

    def _load_distill(self, uri: str | None) -> torch.Tensor:
        if not uri:
            return torch.zeros(self.expected_distill_dim, dtype=torch.float32)
        path = self._resolve_uri(uri)
        if not os.path.exists(path):
            return torch.zeros(self.expected_distill_dim, dtype=torch.float32)
        import numpy as np

        x = np.load(path)
        x = x.squeeze()
        # Ensure fixed size for the stub
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.shape[0] < self.expected_distill_dim:
            pad = self.expected_distill_dim - x.shape[0]
            x = np.pad(x, (0, pad))
        elif x.shape[0] > self.expected_distill_dim:
            x = x[: self.expected_distill_dim]
        return torch.from_numpy(x.astype("float32"))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        eg = self.items[idx]
        wav = self._load_clip(eg["audio_uri"], eg["t0"], eg["t1"], self.sr)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Mel spectrogram -> dB -> [0,1]
        mel = self.melspec(wav)  # [1, M, Tm]
        mel_db = self.amp2db(mel).clamp_(-80, 0)
        mel01 = (mel_db + 80.0) / 80.0
        # Fix time to seg_frames
        Tm = mel01.size(-1)
        if Tm < self.seg_frames:
            pad = self.seg_frames - Tm
            mel01 = torch.nn.functional.pad(mel01, (pad // 2, pad - pad // 2))
        elif Tm > self.seg_frames:
            start = (Tm - self.seg_frames) // 2
            mel01 = mel01[..., start : start + self.seg_frames]

        # Human labels and mask
        y = torch.zeros(len(ALL_DIMS), dtype=torch.float32)
        mask = torch.zeros(len(ALL_DIMS), dtype=torch.float32)
        for d, v in (eg.get("labels") or {}).items():
            if d in DIM_TO_IDX:
                y[DIM_TO_IDX[d]] = float(v)
        for d, m in (eg.get("label_mask") or {}).items():
            if d in DIM_TO_IDX and m:
                mask[DIM_TO_IDX[d]] = 1.0

        # Pseudo labels and confidences
        py = torch.zeros(len(ALL_DIMS), dtype=torch.float32)
        pmask = torch.zeros(len(ALL_DIMS), dtype=torch.float32)
        pwt = torch.zeros(len(ALL_DIMS), dtype=torch.float32)
        for d, v in (eg.get("pseudo_labels") or {}).items():
            if d in DIM_TO_IDX:
                py[DIM_TO_IDX[d]] = float(v)
                pmask[DIM_TO_IDX[d]] = 1.0
        for d, w in (eg.get("pseudo_conf") or {}).items():
            if d in DIM_TO_IDX:
                pwt[DIM_TO_IDX[d]] = float(w)
        # default weight 1 for any pseudo mask that lacks conf
        pwt = torch.where((pmask > 0) & (pwt == 0), torch.ones_like(pwt), pwt)

        # Dataset id
        ds_id = torch.tensor(DS_TO_ID.get(eg.get("dataset", "MAESTRO"), 0), dtype=torch.long)

        # Distillation embedding (optional)
        distill = self._load_distill(eg.get("distill_uri"))  # [expected_distill_dim]

        return mel01, y, mask, ds_id, py, pmask, pwt, distill


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]):
    elems = list(zip(*batch))
    return tuple(torch.stack(e) for e in elems)