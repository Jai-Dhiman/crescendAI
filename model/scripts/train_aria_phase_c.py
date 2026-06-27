"""Aria LoRA Phase C baseline on T1 PercePiano -- LOCAL Apple-Silicon (MPS) runner.

This is the local-Mac counterpart to ``model/jobs/train_aria.py`` (the cloud A100
entrypoint, issue #78 / epic #71). The cloud job clones the repo and stages data from
R2; here the data is already local, so the git-clone + R2-staging bootstrap is stripped
and the accelerator is pinned to MPS.

The model assembly is reused VERBATIM from the proven smoke test
(``scripts/smoke_test_aria.py``) and ``model_improvement/aria_encoder.py``:

    load_composite_labels -> AriaMidiPairDataset(midi_dir=<T1 AMT MIDI>, ...) ->
    DataLoader(collate=aria_midi_collate_fn, pin_memory=False) ->
    AriaLoRAModel(lora_rank=32, layers 8-15) -> train_model(...)

Two corrections / additions relative to the cloud entrypoint:

  1. FOLD SEMANTICS. ``folds.json`` (post-#73) stores SEGMENT IDs in each fold's
     train/val lists, not piece names. The cloud script's
     ``val_pieces = set(folds[i]["val"])`` intersected against ``piece_mapping`` keys
     yields the empty set on this file, silently producing an empty val split. This
     runner maps fold entries directly as segment keys (verified piece-disjoint:
     zero piece straddles a fold's train/val boundary, so within-piece pairs never leak).

  2. MEMORY GUARD (required: prior MPS runs grew memory unbounded). Two layers:
       a. An in-loop callback that calls ``gc.collect() + torch.mps.empty_cache()``
          every N steps (mirrors ``release_accelerator_memory`` in
          ``apps/inference/extract_amt_midi.py``).
       b. A SEPARATE watchdog process (GIL-immune; survives long MPS kernels) that
          samples system-available RAM and trainer RSS every few seconds and SIGKILLs
          the trainer if a configurable floor is breached for N consecutive samples.
          Frequent checkpointing (``--ckpt-every-n-steps``) bounds the loss from a kill.

Data lives in the PRIMARY checkout (``model/data`` is gitignored, absent from the
worktree), so all data paths default to absolute primary-checkout locations.

Usage (run from anywhere; uses the primary venv):

    uv run --project /Users/jdhiman/Documents/crescendai/model python \
        /Users/jdhiman/Documents/crescendai/.worktrees/issue-78-aria-phase-c/model/scripts/train_aria_phase_c.py \
        --folds 0 --max-epochs 25 --batch-size 4

Acceptance (#78): clean MPS convergence (loss down, no NaN) and per-fold + mean pairwise
accuracy beating the frozen-probe baseline of 59.6%. Reports pairwise + R2; saves best
checkpoint per fold under ``data/checkpoints/aria_phase_c_t1/fold_{i}``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence

# Import the worktree's own source (this file lives in <worktree>/model/scripts/),
# so model_improvement / paths resolve to the edited worktree tree, not the primary.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_improvement.aria_encoder import (  # noqa: E402
    AriaLoRAModel,
    AriaMidiPairDataset,
    _get_tokenizer,
    aria_midi_collate_fn,
)
from model_improvement.taxonomy import (  # noqa: E402
    NUM_DIMS,
    load_composite_labels,
)
from model_improvement.training import train_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_aria_phase_c")

# Primary-checkout data root (model/data is gitignored, absent from the worktree).
PRIMARY_DATA = Path("/Users/jdhiman/Documents/crescendai/model/data")

DIM_NAMES = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]

# Ambiguity threshold for pairwise accuracy (matches AriaLoRAModel.on_validation_epoch_end).
AMBIGUOUS_THRESHOLD = 0.05


# --------------------------------------------------------------------------------------
# Memory guard
# --------------------------------------------------------------------------------------

# Watchdog runs as an INDEPENDENT process (own interpreter, own GIL) so it keeps sampling
# even while the trainer is blocked inside a long MPS kernel with the GIL held. It uses
# psutil for both readings: psutil.virtual_memory().available is derived from the same
# Mach VM statistics that `vm_stat`/`memory_pressure` expose, but is more robust to parse
# than shell output; psutil.Process(pid).memory_info().rss gives per-process RSS (which
# `vm_stat` cannot). On breach it logs the reason + last reading, then SIGKILLs the trainer.
_WATCHDOG_SRC = r'''
import os, sys, time, signal
import psutil

pid          = int(sys.argv[1])
free_floor   = float(sys.argv[2])   # bytes; 0 disables
rss_cap      = float(sys.argv[3])   # bytes; 0 disables
interval     = float(sys.argv[4])   # seconds
consecutive  = int(sys.argv[5])
log_path     = sys.argv[6]
swap_growth  = float(sys.argv[7])   # bytes of NEW swap above baseline; 0 disables

def emit(msg):
    line = f"[watchdog] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass

swap_base = psutil.swap_memory().used
emit(f"started pid={pid} free_floor={free_floor/1e9:.2f}GB rss_cap={rss_cap/1e9:.2f}GB "
     f"swap_growth_cap={swap_growth/1e9:.2f}GB swap_base={swap_base/1e9:.2f}GB "
     f"interval={interval}s consecutive={consecutive}")

# RSS is reported near-zero for MPS processes (unified memory is not counted in RSS), so
# the rss_cap is a weak signal here; the system-available floor catches a hard OOM, and the
# swap-growth check catches the THRASH failure mode (macOS keeps "available" above the floor
# by compressing/swapping pages while throughput collapses -- observed in the batch-8 run).
free_breaches = 0
rss_breaches = 0
swap_breaches = 0
while True:
    time.sleep(interval)
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
    except psutil.NoSuchProcess:
        emit("trainer process gone; exiting cleanly")
        sys.exit(0)
    avail = psutil.virtual_memory().available
    swap_now = psutil.swap_memory().used
    swap_delta = swap_now - swap_base

    free_breaches = free_breaches + 1 if (free_floor > 0 and avail < free_floor) else 0
    rss_breaches = rss_breaches + 1 if (rss_cap > 0 and rss > rss_cap) else 0
    swap_breaches = swap_breaches + 1 if (swap_growth > 0 and swap_delta > swap_growth) else 0

    if free_breaches >= consecutive:
        emit(f"KILL: system available RAM {avail/1e9:.2f}GB < floor {free_floor/1e9:.2f}GB "
             f"for {free_breaches} consecutive samples (RSS {rss/1e9:.2f}GB swap+{swap_delta/1e9:.2f}GB)")
        os.kill(pid, signal.SIGKILL)
        sys.exit(1)
    if swap_breaches >= consecutive:
        emit(f"KILL: swap grew +{swap_delta/1e9:.2f}GB > cap {swap_growth/1e9:.2f}GB "
             f"for {swap_breaches} consecutive samples (avail {avail/1e9:.2f}GB) -- thrashing")
        os.kill(pid, signal.SIGKILL)
        sys.exit(1)
    if rss_breaches >= consecutive:
        emit(f"KILL: trainer RSS {rss/1e9:.2f}GB > cap {rss_cap/1e9:.2f}GB "
             f"for {rss_breaches} consecutive samples (system avail {avail/1e9:.2f}GB)")
        os.kill(pid, signal.SIGKILL)
        sys.exit(1)
'''


def start_watchdog(
    free_floor_gb: float,
    rss_cap_gb: float,
    interval_s: float,
    consecutive: int,
    log_path: Path,
    swap_growth_gb: float,
) -> subprocess.Popen:
    """Launch the independent watchdog process targeting THIS process (the trainer)."""
    args = [
        sys.executable, "-c", _WATCHDOG_SRC,
        str(os.getpid()),
        str(free_floor_gb * 1e9),
        str(rss_cap_gb * 1e9),
        str(interval_s),
        str(consecutive),
        str(log_path),
        str(swap_growth_gb * 1e9),
    ]
    proc = subprocess.Popen(args)
    logger.info(
        "watchdog started (pid=%d) floor=%.1fGB rss_cap=%s swap_growth_cap=%s "
        "interval=%.1fs consecutive=%d",
        proc.pid, free_floor_gb,
        f"{rss_cap_gb:.1f}GB" if rss_cap_gb > 0 else "disabled",
        f"{swap_growth_gb:.1f}GB" if swap_growth_gb > 0 else "disabled",
        interval_s, consecutive,
    )
    return proc


class MpsMemoryGuardCallback(pl.Callback):
    """Bound the MPS caching allocator: gc.collect + torch.mps.empty_cache every N steps.

    Mirrors release_accelerator_memory() in apps/inference/extract_amt_midi.py. Also logs
    trainer RSS + system-available RAM periodically so the operator can watch the first few
    steps before walking away.
    """

    def __init__(self, every_n_steps: int) -> None:
        super().__init__()
        self.every_n_steps = max(1, every_n_steps)
        try:
            import psutil
            self._proc = psutil.Process()
            self._psutil = psutil
        except ImportError:
            self._proc = None
            self._psutil = None

    def _release(self) -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (batch_idx + 1) % self.every_n_steps != 0:
            return
        self._release()
        if self._proc is not None and self._psutil is not None:
            rss = self._proc.memory_info().rss / 1e9
            avail = self._psutil.virtual_memory().available / 1e9
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float("nan")
            logger.info(
                "step %d  loss=%.4f  RSS=%.2fGB  sys_avail=%.2fGB  (cache cleared)",
                batch_idx + 1, loss, rss, avail,
            )


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def build_loader(
    keys: list[str],
    midi_dir: Path,
    labels: dict[str, list[float]],
    piece_mapping: dict[str, list[str]],
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Build a within-piece pair DataLoader over the given segment keys (MPS: pin_memory=False)."""
    ds = AriaMidiPairDataset(
        midi_dir=midi_dir,
        labels=labels,
        piece_to_keys=piece_mapping,
        keys=[k for k in keys if k in labels],
        max_seq_len=max_seq_len,
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"0 pairs from {midi_dir} -- AMT MIDI stems must match label keys."
        )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=aria_midi_collate_fn,
        num_workers=0,        # tokenization is cached in __init__; workers add fork risk on MPS
        pin_memory=False,     # pin_memory is CUDA-only; MPS requires False
    )


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def eval_pairwise(model: AriaLoRAModel, loader, device: torch.device) -> float:
    """Mean pairwise ranking accuracy over non-ambiguous (|la-lb|>=thr) pairs, all dims."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        logits = model(
            batch["tokens_a"].to(device), batch["eos_pos_a"].to(device),
            batch["tokens_b"].to(device), batch["eos_pos_b"].to(device),
        )["ranking_logits"].cpu()
        la = batch["labels_a"]
        lb = batch["labels_b"]
        true_rank = (la > lb).float()
        pred_rank = (logits > 0).float()
        non_amb = (la - lb).abs() >= AMBIGUOUS_THRESHOLD
        if non_amb.any():
            correct += (pred_rank[non_amb] == true_rank[non_amb]).sum().item()
            total += int(non_amb.sum().item())
    return correct / total if total > 0 else float("nan")


@torch.no_grad()
def eval_regression_r2(
    model: AriaLoRAModel,
    dataset: AriaMidiPairDataset,
    labels: dict[str, list[float]],
    device: torch.device,
    batch_size: int,
) -> dict:
    """Per-segment R2 of the regression head vs composite labels (per-dim + mean).

    The dataset is pairwise; R2 needs unique segments, so we run the regression head over
    each distinct val segment the dataset tokenized (dataset._tokens), batched.
    """
    model.eval()
    tokenizer = _get_tokenizer()
    pad_id = tokenizer.encode([tokenizer.pad_tok])[0]

    seg_ids = sorted(dataset._tokens.keys())
    if not seg_ids:
        return {"per_dim": {}, "mean": float("nan"), "overall": float("nan"), "n": 0}

    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for i in range(0, len(seg_ids), batch_size):
        chunk = seg_ids[i : i + batch_size]
        seqs = [torch.tensor(dataset._tokens[s], dtype=torch.long) for s in chunk]
        tokens = pad_sequence(seqs, batch_first=True, padding_value=pad_id).to(device)
        eos_pos = torch.tensor([dataset._eos_pos[s] for s in chunk], dtype=torch.long).to(device)
        z = model._encode(tokens, eos_pos)
        scores = model.regression_head(z).cpu().numpy()
        preds.append(scores)
        trues.append(np.array([labels[s] for s in chunk], dtype=np.float32))

    y_pred = np.concatenate(preds, axis=0)   # [N, 6]
    y_true = np.concatenate(trues, axis=0)   # [N, 6]

    per_dim = {}
    r2s = []
    for d in range(NUM_DIMS):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        per_dim[DIM_NAMES[d]] = r2
        if r2 == r2:  # not NaN
            r2s.append(r2)

    ss_res_all = float(np.sum((y_true - y_pred) ** 2))
    ss_tot_all = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    overall = 1.0 - ss_res_all / ss_tot_all if ss_tot_all > 0 else float("nan")

    return {
        "per_dim": per_dim,
        "mean": float(np.mean(r2s)) if r2s else float("nan"),
        "overall": overall,
        "n": len(seg_ids),
    }


# --------------------------------------------------------------------------------------
# Per-fold training
# --------------------------------------------------------------------------------------

def run_fold(args, fold_idx: int, labels, piece_mapping, folds, watchdog_log: Path) -> dict:
    pl.seed_everything(args.seed, workers=True)

    train_keys = list(folds[fold_idx]["train"])
    val_keys = list(folds[fold_idx]["val"])
    logger.info(
        "fold %d: %d train segs, %d val segs (mapped to labels)",
        fold_idx, len(train_keys), len(val_keys),
    )

    train_loader = build_loader(
        train_keys, args.midi_dir, labels, piece_mapping,
        args.max_seq_len, args.batch_size, shuffle=True,
    )
    val_loader = build_loader(
        val_keys, args.midi_dir, labels, piece_mapping,
        args.max_seq_len, args.batch_size, shuffle=False,
    )
    logger.info(
        "fold %d: %d train pairs, %d val pairs",
        fold_idx, len(train_loader.dataset), len(val_loader.dataset),
    )

    model = AriaLoRAModel(
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    ckpt_dir = Path(args.checkpoint_dir) / args.model_name / f"fold_{fold_idx}"

    extra_callbacks: list[pl.Callback] = [
        MpsMemoryGuardCallback(every_n_steps=args.empty_cache_every),
    ]
    if args.ckpt_every_n_steps > 0:
        # Frequent intra-epoch checkpoint so a watchdog SIGKILL loses little progress.
        extra_callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir / "periodic"),
                filename="periodic-{step}",
                every_n_train_steps=args.ckpt_every_n_steps,
                save_top_k=1,
                monitor=None,
            )
        )

    # Trackio always on (logs to the local crescendai-training db; persists to an HF Space
    # only if TRACKIO_SPACE_ID is set, which TrackioCallback reads).
    if args.trackio_space_id:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space_id
    trackio_id = f"aria-phase-c-fold{fold_idx}"

    t0 = time.time()
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=args.model_name,
        fold_idx=fold_idx,
        checkpoint_dir=str(args.checkpoint_dir),
        max_epochs=args.max_epochs,
        monitor="val_loss",
        patience=args.patience,
        accelerator="mps",
        trackio_experiment_id=trackio_id,
        extra_callbacks=extra_callbacks,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else None,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else None,
    )
    train_secs = time.time() - t0

    # Load best checkpoint (by val_loss) for a clean eval pass.
    best_path = trainer.checkpoint_callback.best_model_path
    device = torch.device("mps")
    if best_path and Path(best_path).exists():
        logger.info("fold %d: loading best checkpoint %s", fold_idx, best_path)
        best = AriaLoRAModel.load_from_checkpoint(best_path, map_location=device)
    else:
        logger.warning("fold %d: no best checkpoint found; evaluating final model", fold_idx)
        best = model
    best = best.to(device)

    # Final eval is forward-only, so it tolerates a larger batch than training. Reuse the
    # already-tokenized val dataset (avoid re-tokenizing) with a bigger eval batch.
    eval_loader = torch.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=aria_midi_collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    pairwise = eval_pairwise(best, eval_loader, device)
    r2 = eval_regression_r2(best, val_loader.dataset, labels, device, args.eval_batch_size)

    # best val_pairwise_acc observed during training (logged each val epoch)
    logged_acc = trainer.callback_metrics.get("val_pairwise_acc")
    logged_acc = float(logged_acc) if logged_acc is not None else float("nan")

    result = {
        "fold": fold_idx,
        "best_checkpoint": best_path,
        "train_secs": round(train_secs, 1),
        "epochs_run": trainer.current_epoch + 1,
        "n_train_pairs": len(train_loader.dataset),
        "n_val_pairs": len(val_loader.dataset),
        "pairwise_acc": pairwise,
        "pairwise_acc_logged_last": logged_acc,
        "r2_mean": r2["mean"],
        "r2_overall": r2["overall"],
        "r2_per_dim": r2["per_dim"],
        "r2_n_segments": r2["n"],
    }
    logger.info(
        "fold %d DONE: pairwise=%.4f  r2_mean=%.4f  (%.1f min, %d epochs)",
        fold_idx, pairwise, r2["mean"], train_secs / 60, result["epochs_run"],
    )

    # Free fold memory before the next fold.
    del model, best, trainer, train_loader, val_loader
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aria LoRA Phase C baseline on T1 (local MPS).")
    # Data (absolute defaults -> primary checkout; model/data is gitignored, absent here)
    p.add_argument("--midi-dir", type=Path, default=PRIMARY_DATA / "midi" / "amt" / "t1",
                   help="T1 AMT performance MIDI dir (z_perf input).")
    p.add_argument("--labels-path", type=Path,
                   default=PRIMARY_DATA / "labels" / "composite" / "composite_labels.json")
    p.add_argument("--folds-path", type=Path,
                   default=PRIMARY_DATA / "labels" / "percepiano" / "folds.json")
    p.add_argument("--piece-mapping-path", type=Path,
                   default=PRIMARY_DATA / "labels" / "percepiano" / "piece_mapping.json")
    p.add_argument("--checkpoint-dir", type=Path, default=PRIMARY_DATA / "checkpoints")
    p.add_argument("--results-path", type=Path,
                   default=PRIMARY_DATA / "results" / "aria_phase_c_t1.json")
    p.add_argument("--model-name", default="aria_phase_c_t1")
    p.add_argument("--trackio-space-id", default=os.environ.get("TRACKIO_SPACE_ID"),
                   help="HF Space id for persistent Trackio dashboard (sets TRACKIO_SPACE_ID).")
    # Training
    p.add_argument("--folds", default="0,1,2,3", help="Comma-separated fold indices.")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Train batch size. 2 is the MPS sweet spot (batch 4+ thrashes/swaps).")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Forward-only eval batch (tolerates larger than train batch).")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    # Per-epoch subsampling: 8358 within-piece pairs/epoch is ~5h/epoch on MPS. Cap the
    # number of (reshuffled) train/val batches per epoch. 0 = use all. The FINAL eval still
    # runs over the FULL val set, so the reported metric is not subsampled.
    p.add_argument("--limit-train-batches", type=int, default=400,
                   help="Max train batches per epoch (reshuffled each epoch). 0 = all.")
    p.add_argument("--limit-val-batches", type=int, default=150,
                   help="Max val batches per epoch for the early-stopping signal. 0 = all.")
    # Checkpoint cadence
    p.add_argument("--ckpt-every-n-steps", type=int, default=200,
                   help="Intra-epoch checkpoint cadence (0 disables).")
    # Memory guard
    p.add_argument("--empty-cache-every", type=int, default=20,
                   help="Steps between gc.collect + mps.empty_cache (and a RSS log line).")
    p.add_argument("--mem-free-floor-gb", type=float,
                   default=float(os.environ.get("MEM_FREE_FLOOR_GB", "2.0")),
                   help="Watchdog: SIGKILL if system available RAM < this for N samples.")
    p.add_argument("--mem-rss-cap-gb", type=float,
                   default=float(os.environ.get("MEM_RSS_CAP_GB", "0")),
                   help="Watchdog: SIGKILL if trainer RSS > this for N samples (0 disables).")
    p.add_argument("--mem-swap-growth-gb", type=float,
                   default=float(os.environ.get("MEM_SWAP_GROWTH_GB", "8.0")),
                   help="Watchdog: SIGKILL if swap grows > this above startup baseline for N "
                        "samples (catches MPS thrash the free-RAM floor misses). 0 disables.")
    p.add_argument("--mem-sample-interval", type=float,
                   default=float(os.environ.get("MEM_SAMPLE_INTERVAL", "3.0")))
    p.add_argument("--mem-consecutive", type=int,
                   default=int(os.environ.get("MEM_CONSECUTIVE", "3")))
    p.add_argument("--no-watchdog", action="store_true", help="Disable the watchdog process.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for pth in (args.midi_dir, args.labels_path, args.folds_path, args.piece_mapping_path):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Required data path missing: {pth}")
    args.results_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available -- this runner targets Apple Silicon (MPS).")

    labels_raw = load_composite_labels(args.labels_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}
    piece_mapping = json.loads(args.piece_mapping_path.read_text())
    folds = json.loads(args.folds_path.read_text())
    if not isinstance(folds, list):
        raise ValueError(f"Expected folds.json to be a list of folds, got {type(folds)}")

    fold_idxs = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    logger.info("Running folds %s of %d available", fold_idxs, len(folds))

    watchdog_log = args.results_path.with_suffix(".watchdog.log")
    watchdog = None
    if not args.no_watchdog:
        watchdog = start_watchdog(
            args.mem_free_floor_gb, args.mem_rss_cap_gb,
            args.mem_sample_interval, args.mem_consecutive, watchdog_log,
            args.mem_swap_growth_gb,
        )

    results = []
    try:
        for fi in fold_idxs:
            if fi < 0 or fi >= len(folds):
                raise IndexError(f"fold {fi} out of range (0..{len(folds) - 1})")
            results.append(run_fold(args, fi, labels, piece_mapping, folds, watchdog_log))
            # Persist incrementally so a later kill keeps earlier folds.
            _write_results(args, results)
    finally:
        if watchdog is not None and watchdog.poll() is None:
            watchdog.terminate()
            logger.info("watchdog terminated")

    _summarize(args, results)


def _write_results(args, results: list[dict]) -> None:
    # Merge with any folds already on disk (one-process-per-fold runs accumulate here),
    # replacing same-index folds from this run and keeping the rest.
    merged: dict[int, dict] = {}
    if args.results_path.exists():
        try:
            prior = json.loads(args.results_path.read_text())
            for r in prior.get("folds", []):
                merged[r["fold"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    for r in results:
        merged[r["fold"]] = r
    all_folds = [merged[k] for k in sorted(merged)]

    pairwise = [r["pairwise_acc"] for r in all_folds if r["pairwise_acc"] == r["pairwise_acc"]]
    r2means = [r["r2_mean"] for r in all_folds if r["r2_mean"] == r["r2_mean"]]
    payload = {
        "model_name": args.model_name,
        "baseline_frozen_probe_pairwise": 0.596,
        "config": {
            "lora_rank": args.lora_rank,
            "lora_layers": "8-15",
            "max_epochs": args.max_epochs,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "midi_dir": str(args.midi_dir),
            "note": "T1 AMT MIDI is a fluidsynth timbre proxy -> treat as LOWER BOUND.",
        },
        "folds": all_folds,
        "mean_pairwise_acc": float(np.mean(pairwise)) if pairwise else None,
        "mean_r2": float(np.mean(r2means)) if r2means else None,
    }
    args.results_path.write_text(json.dumps(payload, indent=2))


def _summarize(args, results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Aria Phase C baseline on T1 (PercePiano) -- {len(results)} fold(s)")
    print(f"Frozen-probe baseline to beat: 0.596 pairwise")
    print("-" * 72)
    print(f"{'fold':>4}  {'pairwise':>9}  {'r2_mean':>8}  {'r2_overall':>10}  {'epochs':>6}  {'min':>6}")
    for r in results:
        print(
            f"{r['fold']:>4}  {r['pairwise_acc']:>9.4f}  {r['r2_mean']:>8.4f}  "
            f"{r['r2_overall']:>10.4f}  {r['epochs_run']:>6}  {r['train_secs'] / 60:>6.1f}"
        )
    print("-" * 72)
    pairwise = [r["pairwise_acc"] for r in results if r["pairwise_acc"] == r["pairwise_acc"]]
    r2means = [r["r2_mean"] for r in results if r["r2_mean"] == r["r2_mean"]]
    if pairwise:
        mp = float(np.mean(pairwise))
        verdict = "PASS (beats 0.596)" if mp > 0.596 else "BELOW baseline"
        print(f"MEAN pairwise = {mp:.4f}  [{verdict}]")
    if r2means:
        print(f"MEAN r2       = {float(np.mean(r2means)):.4f}")
    print(f"Results JSON  -> {args.results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
