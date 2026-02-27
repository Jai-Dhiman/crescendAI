"""Quick diagnostic: test DataLoader + forward pass + 1-epoch fit for S1.

Run from model/ directory:
    python debug_symbolic.py
"""
import sys
import time
import json
import torch
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from model_improvement.symbolic_encoders import TransformerSymbolicEncoder
from model_improvement.tokenizer import PianoTokenizer
from model_improvement.data import PairedPerformanceDataset, symbolic_collate_fn
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS
from model_improvement.training import train_model

DATA_DIR = Path("data")
cache_dir = DATA_DIR / "percepiano_cache"

# --- Load data ---
print("[1/7] Loading labels...")
labels_raw = load_composite_labels(DATA_DIR / "composite_labels" / "composite_labels.json")
labels = {k: v.tolist() for k, v in labels_raw.items()}
print(f"  {len(labels)} labels loaded")

print("[2/7] Loading tokens...")
all_tokens = torch.load(
    DATA_DIR / "pretrain_cache" / "tokens" / "all_tokens.pt",
    map_location="cpu", weights_only=False,
)
token_sequences = {
    k.replace("percepiano__", ""): v
    for k, v in all_tokens.items()
    if k.startswith("percepiano__")
}
print(f"  {len(token_sequences)} PercePiano token sequences")

print("[3/7] Loading fold + piece mapping...")
with open(cache_dir / "folds.json") as f:
    folds = json.load(f)
with open(cache_dir / "piece_mapping.json") as f:
    piece_to_keys = json.load(f)

fold = folds[0]
train_keys = [k for k in fold["train"] if k in token_sequences]
val_keys = [k for k in fold["val"] if k in token_sequences]
print(f"  Fold 0: {len(train_keys)} train, {len(val_keys)} val")

# --- Check key overlap ---
print("[4/7] Checking key overlap (dataset vs token_sequences)...")
ds = PairedPerformanceDataset(
    cache_dir=cache_dir, labels=labels,
    piece_to_keys=piece_to_keys, keys=train_keys,
)
val_ds = PairedPerformanceDataset(
    cache_dir=cache_dir, labels=labels,
    piece_to_keys=piece_to_keys, keys=val_keys,
)
sample = ds[0]
key_a, key_b = sample["key_a"], sample["key_b"]
print(f"  Sample pair: key_a={key_a!r}, key_b={key_b!r}")
print(f"  key_a in token_sequences: {key_a in token_sequences}")
print(f"  key_b in token_sequences: {key_b in token_sequences}")

# --- Test DataLoader ---
print("[5/7] Testing DataLoader iteration (train + val)...")
collate = partial(symbolic_collate_fn, token_sequences=token_sequences)
train_loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=0)

t0 = time.time()
batch = next(iter(train_loader))
print(f"  Train batch in {time.time() - t0:.2f}s, keys: {list(batch.keys())}, shape: {batch['input_ids_a'].shape}")

t0 = time.time()
val_batch = next(iter(val_loader))
print(f"  Val batch in {time.time() - t0:.2f}s, keys: {list(val_batch.keys())}, shape: {val_batch['input_ids_a'].shape}")

# --- Test single forward pass ---
tokenizer = PianoTokenizer(max_seq_len=2048)
S1_CONFIG = {
    "vocab_size": tokenizer.vocab_size + 1,
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "hidden_dim": 512,
    "num_labels": NUM_DIMS,
}

print("[6/7] Testing single forward pass (eager mode, no torch.compile)...")
model = TransformerSymbolicEncoder(**S1_CONFIG, stage="finetune")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Use batch_size=2 locally (full batch_size=32 needs ~26GB for attention alone)
small_batch = {k: v[:2].to(device) if hasattr(v, "to") else v for k, v in batch.items()}
print(f"  Using batch_size=2 for local test (shape: {small_batch['input_ids_a'].shape})")

t0 = time.time()
with torch.no_grad():
    out = model.training_step(small_batch, 0)
print(f"  training_step completed in {time.time() - t0:.2f}s, loss={out}")

# --- Test 1-epoch Lightning fit ---
print("[7/7] Testing 1-epoch Lightning trainer.fit (batch_size=2 for local)...")
model = TransformerSymbolicEncoder(**S1_CONFIG, stage="finetune")

# Rebuild loaders with batch_size=2 for local memory constraints
small_train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)
small_val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate, num_workers=0)

t0 = time.time()
trainer = train_model(
    model, small_train_loader, small_val_loader,
    "S1_debug", fold_idx=0,
    checkpoint_dir=Path("/tmp/crescendai_debug_ckpt"),
    max_epochs=1,
)
print(f"  trainer.fit completed in {time.time() - t0:.2f}s")
val_loss = trainer.callback_metrics.get("val_loss", "N/A")
val_acc = trainer.callback_metrics.get("val_pairwise_acc", "N/A")
print(f"  val_loss={val_loss}, val_pairwise_acc={val_acc}")

print("\nAll checks passed! Training pipeline is working.")
