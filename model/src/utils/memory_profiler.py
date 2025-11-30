"""
Memory profiling utilities for debugging memory leaks during training.

Usage:
    from src.utils.memory_profiler import MemoryProfilerCallback, log_memory

    # Add to trainer
    trainer = pl.Trainer(
        callbacks=[MemoryProfilerCallback(log_every_n_steps=50)],
        ...
    )
"""

import gc
import os
import psutil
import torch
import pytorch_lightning as pl
from typing import Any, Optional
from datetime import datetime


def get_memory_stats() -> dict:
    """Get current memory statistics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    stats = {
        "cpu_rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
        "cpu_vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        "cpu_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return stats


def log_memory(tag: str = "", verbose: bool = True) -> dict:
    """Log current memory usage with a tag."""
    stats = get_memory_stats()

    if verbose:
        gpu_str = ""
        if "gpu_allocated_mb" in stats:
            gpu_str = f", GPU: {stats['gpu_allocated_mb']:.1f}MB"

        print(f"[MEM {tag}] CPU: {stats['cpu_rss_mb']:.1f}MB ({stats['cpu_percent']:.1f}%){gpu_str}")

    return stats


class MemoryProfilerCallback(pl.Callback):
    """
    PyTorch Lightning callback for detailed memory profiling.

    Tracks memory at:
    - Start/end of each batch
    - Before/after forward pass
    - Before/after backward pass
    - Before/after optimizer step

    Logs deltas to identify which operation is leaking.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_to_file: Optional[str] = None,
        detailed: bool = True,
        warn_threshold_mb: float = 100.0,
    ):
        """
        Args:
            log_every_n_steps: Log memory every N training steps
            log_to_file: Optional file path to write detailed logs
            detailed: If True, log per-operation breakdown
            warn_threshold_mb: Warn if memory grows by more than this in one step
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_to_file = log_to_file
        self.detailed = detailed
        self.warn_threshold_mb = warn_threshold_mb

        # Tracking
        self.initial_memory = None
        self.last_memory = None
        self.step_count = 0
        self.memory_history = []

        # Per-step tracking
        self.step_start_memory = None

        # File handle
        self.file_handle = None

    def _log(self, message: str):
        """Log to console and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)

        if self.file_handle:
            self.file_handle.write(full_message + "\n")
            self.file_handle.flush()

    def on_train_start(self, trainer, pl_module):
        """Called at the start of training."""
        if self.log_to_file:
            self.file_handle = open(self.log_to_file, "w")

        self.initial_memory = get_memory_stats()
        self.last_memory = self.initial_memory.copy()

        self._log("=" * 70)
        self._log("MEMORY PROFILER STARTED")
        self._log("=" * 70)
        self._log(f"Initial CPU: {self.initial_memory['cpu_rss_mb']:.1f}MB ({self.initial_memory['cpu_percent']:.1f}%)")
        if "gpu_allocated_mb" in self.initial_memory:
            self._log(f"Initial GPU: {self.initial_memory['gpu_allocated_mb']:.1f}MB")
        self._log("")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called at the start of each training batch."""
        self.step_start_memory = get_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        self.step_count += 1
        current = get_memory_stats()

        # Calculate deltas
        step_delta = current["cpu_rss_mb"] - self.step_start_memory["cpu_rss_mb"]
        total_delta = current["cpu_rss_mb"] - self.initial_memory["cpu_rss_mb"]

        # Store history (capped to prevent memory leak)
        # Keep last 200 entries, trim to 100 when exceeded
        self.memory_history.append({
            "step": self.step_count,
            "batch_idx": batch_idx,
            "cpu_mb": current["cpu_rss_mb"],
            "step_delta_mb": step_delta,
            "total_delta_mb": total_delta,
        })
        if len(self.memory_history) > 200:
            self.memory_history = self.memory_history[-100:]

        # Log periodically
        should_log = (
            self.step_count % self.log_every_n_steps == 0 or
            abs(step_delta) > self.warn_threshold_mb
        )

        if should_log:
            self._log(
                f"Step {self.step_count:5d} | "
                f"CPU: {current['cpu_rss_mb']:.1f}MB ({current['cpu_percent']:.1f}%) | "
                f"Step: {step_delta:+.1f}MB | "
                f"Total: {total_delta:+.1f}MB"
            )

            if abs(step_delta) > self.warn_threshold_mb:
                self._log(f"  WARNING: Large memory change in single step!")

        self.last_memory = current

    def on_validation_start(self, trainer, pl_module):
        """Called at the start of validation."""
        current = get_memory_stats()
        self._log(f"\n--- VALIDATION START | CPU: {current['cpu_rss_mb']:.1f}MB ---")

    def on_validation_end(self, trainer, pl_module):
        """Called at the end of validation."""
        current = get_memory_stats()
        self._log(f"--- VALIDATION END | CPU: {current['cpu_rss_mb']:.1f}MB ---\n")

        # Force garbage collection after validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        after_gc = get_memory_stats()
        freed = current["cpu_rss_mb"] - after_gc["cpu_rss_mb"]
        if freed > 10:
            self._log(f"  GC freed {freed:.1f}MB")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        current = get_memory_stats()
        total_delta = current["cpu_rss_mb"] - self.initial_memory["cpu_rss_mb"]

        self._log("")
        self._log("=" * 70)
        self._log(f"EPOCH {trainer.current_epoch} COMPLETE")
        self._log("=" * 70)
        self._log(f"Current CPU: {current['cpu_rss_mb']:.1f}MB ({current['cpu_percent']:.1f}%)")
        self._log(f"Total growth since start: {total_delta:+.1f}MB")

        # Analyze growth pattern
        if len(self.memory_history) > 10:
            recent = self.memory_history[-100:]
            avg_step_delta = sum(h["step_delta_mb"] for h in recent) / len(recent)
            self._log(f"Average growth per step (last 100): {avg_step_delta:.3f}MB")

            if avg_step_delta > 0.5:
                self._log("WARNING: Memory is growing steadily - likely leak!")
                projected = avg_step_delta * 7119  # Full epoch
                self._log(f"Projected growth for full epoch: {projected:.1f}MB")

        self._log("")

        # Force GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training."""
        current = get_memory_stats()
        total_delta = current["cpu_rss_mb"] - self.initial_memory["cpu_rss_mb"]

        self._log("")
        self._log("=" * 70)
        self._log("TRAINING COMPLETE - MEMORY SUMMARY")
        self._log("=" * 70)
        self._log(f"Initial CPU: {self.initial_memory['cpu_rss_mb']:.1f}MB")
        self._log(f"Final CPU: {current['cpu_rss_mb']:.1f}MB")
        self._log(f"Total growth: {total_delta:+.1f}MB over {self.step_count} steps")
        self._log(f"Average per step: {total_delta / max(self.step_count, 1):.3f}MB")

        if self.file_handle:
            self.file_handle.close()


class DetailedMemoryProfiler:
    """
    Context manager for profiling specific code blocks.

    Usage:
        with DetailedMemoryProfiler("forward_pass"):
            output = model(x)
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_stats = None

    def __enter__(self):
        if self.enabled:
            gc.collect()
            self.start_stats = get_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_stats:
            end_stats = get_memory_stats()
            delta = end_stats["cpu_rss_mb"] - self.start_stats["cpu_rss_mb"]

            if abs(delta) > 10:  # Only log if significant
                print(f"  [{self.name}] Memory delta: {delta:+.1f}MB")

        return False


def profile_object_counts(top_n: int = 10):
    """Profile Python object counts to find what's accumulating."""
    import sys
    from collections import Counter

    gc.collect()

    counts = Counter()
    for obj in gc.get_objects():
        counts[type(obj).__name__] += 1

    print("\nTop object types by count:")
    for obj_type, count in counts.most_common(top_n):
        print(f"  {obj_type}: {count:,}")

    # Also check tensor counts
    tensor_count = 0
    tensor_size = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            tensor_count += 1
            tensor_size += obj.numel() * obj.element_size()

    print(f"\nTorch tensors: {tensor_count:,} ({tensor_size / 1024 / 1024:.1f}MB)")


if __name__ == "__main__":
    # Test the profiler
    print("Memory Profiler Test")
    print("=" * 50)

    stats = log_memory("initial")

    # Allocate some memory
    data = [torch.randn(1000, 1000) for _ in range(10)]
    log_memory("after allocation")

    # Free it
    del data
    gc.collect()
    log_memory("after gc")

    profile_object_counts()
