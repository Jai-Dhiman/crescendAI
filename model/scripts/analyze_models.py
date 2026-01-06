"""
Model Comparison & Fusion Analysis Script.

Compares PercePiano Symbolic (MIDI) vs Audio (MERT) baselines and tests fusion.

Usage:
    cd /Users/jdhiman/Documents/crescendai/model
    uv run python scripts/analyze_models.py

Prerequisites:
    - rclone configured with 'gdrive' remote
    - Audio model checkpoints on GDrive
    - (Optional) Symbolic predictions from Thunder Compute notebook
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import r2_score

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_ROOT = PROJECT_ROOT / "data" / "analysis"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
MERT_CACHE = DATA_ROOT / "mert_embeddings"
LABEL_FILE = DATA_ROOT / "labels" / "label_2round_mean_reg_19_with0_rm_highstd0.json"
FOLD_ASSIGNMENTS_FILE = DATA_ROOT / "audio_fold_assignments.json"
SYMBOLIC_PREDS_FILE = DATA_ROOT / "symbolic_predictions.json"
RESULTS_FILE = DATA_ROOT / "model_comparison_results.json"

DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamic_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "sophistication",
    "interpretation",
]

# Symbolic results from training (hardcoded as reference)
SYMBOLIC_DIM_R2 = {
    "articulation_length": 0.5487,
    "tempo": 0.5329,
    "mood_imagination": 0.4549,
    "articulation_touch": 0.4078,
    "mood_valence": 0.3988,
    "timbre_depth": 0.3961,
    "sophistication": 0.3938,
    "timbre_loudness": 0.3897,
    "pedal_amount": 0.3716,
    "space": 0.3525,
    "mood_energy": 0.3450,
    "interpretation": 0.3239,
    "timbre_variety": 0.3066,
    "balance": 0.2984,
    "drama": 0.2941,
    "timbre_brightness": 0.2454,
    "pedal_clarity": 0.2407,
    "dynamic_range": 0.2257,
    "timing": 0.1254,
}
SYMBOLIC_OVERALL_R2 = 0.3501


def download_data():
    """Download required data from GDrive."""
    print("=" * 60)
    print("DOWNLOADING DATA FROM GDRIVE")
    print("=" * 60)

    for d in [
        CHECKPOINT_ROOT / "audio",
        CHECKPOINT_ROOT / "symbolic",
        MERT_CACHE,
        LABEL_FILE.parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    downloads = [
        (
            "gdrive:crescendai_data/checkpoints/audio_baseline",
            CHECKPOINT_ROOT / "audio",
        ),
        ("gdrive:crescendai_data/percepiano_labels", LABEL_FILE.parent),
        (
            "gdrive:crescendai_data/audio_baseline/audio_fold_assignments.json",
            FOLD_ASSIGNMENTS_FILE,
        ),
        (
            "gdrive:crescendai_data/predictions/symbolic_predictions.json",
            SYMBOLIC_PREDS_FILE,
        ),
    ]

    for src, dst in downloads:
        print(f"  {src.split('/')[-1]}...")
        if dst.is_file() or str(dst).endswith(".json"):
            subprocess.run(["rclone", "copyto", src, str(dst)], capture_output=True)
        else:
            subprocess.run(["rclone", "copy", src, str(dst)], capture_output=True)

    print(
        f"\nAudio checkpoints: {len(list((CHECKPOINT_ROOT / 'audio').glob('*.ckpt')))}"
    )
    print(f"Labels: {LABEL_FILE.exists()}")
    print(f"Fold assignments: {FOLD_ASSIGNMENTS_FILE.exists()}")
    print(f"Symbolic predictions: {SYMBOLIC_PREDS_FILE.exists()}")


def download_mert_embeddings(keys):
    """Download MERT embeddings for specified keys."""
    existing = {p.stem for p in MERT_CACHE.glob("*.pt")}
    to_download = set(keys) - existing

    if not to_download:
        print(f"All {len(keys)} MERT embeddings already cached")
        return

    print(f"Downloading {len(to_download)} MERT embeddings...")
    for i, key in enumerate(to_download):
        src = f"gdrive:crescendai_data/audio_baseline/mert_embeddings/{key}.pt"
        dst = MERT_CACHE / f"{key}.pt"
        subprocess.run(["rclone", "copyto", src, str(dst)], capture_output=True)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(to_download)}")
    print(f"Downloaded {len(to_download)} embeddings")


def load_audio_models(device):
    """Load audio model checkpoints."""
    # Direct import to avoid pulling in unrelated modules
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audio_baseline",
        PROJECT_ROOT / "src" / "percepiano" / "models" / "audio_baseline.py",
    )
    audio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_module)
    AudioPercePianoModel = audio_module.AudioPercePianoModel

    models = {}
    for fold in range(4):
        ckpt_path = CHECKPOINT_ROOT / "audio" / f"fold{fold}_best.ckpt"
        if ckpt_path.exists():
            # Load checkpoint and fix key names (clf -> classifier)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint["state_dict"]

            # Rename keys if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("clf.", "classifier.")
                new_state_dict[new_key] = v

            # Get model hyperparameters (filter out training params)
            hparams = checkpoint.get("hyper_parameters", {})
            model_params = {
                k: v
                for k, v in hparams.items()
                if k
                in [
                    "input_dim",
                    "hidden_dim",
                    "num_labels",
                    "dropout",
                    "learning_rate",
                    "weight_decay",
                    "pooling",
                ]
            }
            model = AudioPercePianoModel(**model_params)
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            model.eval()
            models[fold] = model
    return models


def predict_audio(model, key, device):
    """Generate audio prediction for a single sample."""
    embed_path = MERT_CACHE / f"{key}.pt"
    if not embed_path.exists():
        return None

    embeddings = torch.load(embed_path, weights_only=True)
    if embeddings.shape[0] > 1000:
        embeddings = embeddings[:1000]

    embeddings = embeddings.unsqueeze(0).to(device)
    attention_mask = torch.ones(1, embeddings.shape[1], dtype=torch.bool, device=device)

    with torch.no_grad():
        pred = model(embeddings, attention_mask)

    return pred.squeeze(0).cpu().numpy()


def run_sanity_checks(predictions, labels, keys):
    """Run sanity checks on audio predictions."""
    preds_array = np.array([predictions[k] for k in keys])
    labels_array = np.array([labels[k] for k in keys])

    print("\n" + "=" * 60)
    print("SANITY CHECK 1: Dispersion Analysis (Mean Regression)")
    print("=" * 60)
    print(f"\n{'Dimension':<25} {'Label Std':>10} {'Pred Std':>10} {'Ratio':>8}")
    print("-" * 55)

    ratios = []
    for i, dim in enumerate(DIMENSIONS):
        label_std = labels_array[:, i].std()
        pred_std = preds_array[:, i].std()
        ratio = pred_std / label_std if label_std > 0 else 0
        ratios.append(ratio)
        status = "OK" if ratio >= 0.6 else "LOW" if ratio >= 0.4 else "SEVERE"
        print(
            f"{dim:<25} {label_std:>10.4f} {pred_std:>10.4f} {ratio:>8.2f} [{status}]"
        )

    avg_ratio = np.mean(ratios)
    print(f"\nAverage dispersion ratio: {avg_ratio:.3f}")

    print("\n" + "=" * 60)
    print("SANITY CHECK 2: Energy Correlation")
    print("=" * 60)

    energies = []
    for key in keys:
        embed_path = MERT_CACHE / f"{key}.pt"
        if embed_path.exists():
            embeddings = torch.load(embed_path, weights_only=True)
            energy = embeddings.mean(dim=0).norm().item()
            energies.append(energy)

    energy_array = np.array(energies)
    print(f"\n{'Dimension':<25} {'Corr':>10} {'Concern':>10}")
    print("-" * 50)

    for i, dim in enumerate(DIMENSIONS):
        corr, _ = stats.pearsonr(energy_array, preds_array[:, i])
        concern = "HIGH" if abs(corr) > 0.5 else "MOD" if abs(corr) > 0.3 else ""
        print(f"{dim:<25} {corr:>+10.3f} {concern:>10}")

    return avg_ratio


def run_comparison(audio_dim_r2, audio_overall_r2):
    """Compare audio vs symbolic model."""
    print("\n" + "=" * 70)
    print("AUDIO vs SYMBOLIC COMPARISON")
    print("=" * 70)
    print(
        f"\n{'Dimension':<25} {'Symbolic':>10} {'Audio':>10} {'Winner':>10} {'Delta':>10}"
    )
    print("-" * 70)

    audio_wins = 0
    for dim in DIMENSIONS:
        sym_r2 = SYMBOLIC_DIM_R2.get(dim, 0)
        aud_r2 = audio_dim_r2.get(dim, 0)
        delta = aud_r2 - sym_r2
        winner = "AUDIO" if aud_r2 > sym_r2 else "SYMBOLIC"
        if aud_r2 > sym_r2:
            audio_wins += 1
        print(
            f"{dim:<25} {sym_r2:>+10.4f} {aud_r2:>+10.4f} {winner:>10} {delta:>+10.4f}"
        )

    print("-" * 70)
    print(f"\nAudio wins: {audio_wins}/19")
    print(
        f"Overall: Audio={audio_overall_r2:.4f} vs Symbolic={SYMBOLIC_OVERALL_R2:.4f}"
    )

    return audio_wins


def run_fusion(audio_preds, symbolic_preds, labels, keys):
    """Run fusion testing with aligned predictions."""
    print("\n" + "=" * 70)
    print("FUSION TESTING")
    print("=" * 70)

    audio_arr = np.array([audio_preds[k] for k in keys])
    sym_arr = np.array([np.array(symbolic_preds[k]) for k in keys])
    labels_arr = np.array([labels[k] for k in keys])

    audio_r2 = r2_score(labels_arr, audio_arr)
    sym_r2 = r2_score(labels_arr, sym_arr)

    # Strategy 1: Simple Average
    avg_fusion = (audio_arr + sym_arr) / 2
    avg_r2 = r2_score(labels_arr, avg_fusion)

    # Strategy 2: Optimal Weights
    optimal_weights = {}
    weighted_fusion = np.zeros_like(audio_arr)
    for i, dim in enumerate(DIMENSIONS):
        best_w, best_r2 = 0.5, -np.inf
        for w in np.arange(0, 1.01, 0.05):
            fused = w * audio_arr[:, i] + (1 - w) * sym_arr[:, i]
            r2 = r2_score(labels_arr[:, i], fused)
            if r2 > best_r2:
                best_w, best_r2 = w, r2
        optimal_weights[dim] = best_w
        weighted_fusion[:, i] = best_w * audio_arr[:, i] + (1 - best_w) * sym_arr[:, i]
    weighted_r2 = r2_score(labels_arr, weighted_fusion)

    # Strategy 3: Oracle
    oracle_fusion = np.zeros_like(audio_arr)
    for i, dim in enumerate(DIMENSIONS):
        a_r2 = r2_score(labels_arr[:, i], audio_arr[:, i])
        s_r2 = r2_score(labels_arr[:, i], sym_arr[:, i])
        oracle_fusion[:, i] = audio_arr[:, i] if a_r2 > s_r2 else sym_arr[:, i]
    oracle_r2 = r2_score(labels_arr, oracle_fusion)

    print(f"""
    MODEL                           R2        DELTA
    ----------------------------------------------------------------
    Audio Only                     {audio_r2:.4f}     baseline
    Symbolic Only                  {sym_r2:.4f}    {sym_r2 - audio_r2:+.4f}
    Simple Average Fusion          {avg_r2:.4f}    {avg_r2 - audio_r2:+.4f}
    Optimal Weighted Fusion        {weighted_r2:.4f}    {weighted_r2 - audio_r2:+.4f}
    Oracle Selection (upper bound) {oracle_r2:.4f}    {oracle_r2 - audio_r2:+.4f}
    ----------------------------------------------------------------
    """)

    return {
        "audio_r2": audio_r2,
        "symbolic_r2": sym_r2,
        "simple_fusion_r2": avg_r2,
        "weighted_fusion_r2": weighted_r2,
        "oracle_r2": oracle_r2,
        "optimal_weights": optimal_weights,
    }


def save_scatter_plots(predictions, labels, dim_r2, keys):
    """Save scatter plots for each dimension."""
    preds_arr = np.array([predictions[k] for k in keys])
    labels_arr = np.array([labels[k] for k in keys])

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        ax.scatter(labels_arr[:, i], preds_arr[:, i], alpha=0.3, s=10)
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.axhline(preds_arr[:, i].mean(), color="orange", linestyle=":", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{dim}\nR2={dim_r2[dim]:.3f}")

    axes[-1].set_visible(False)
    plt.tight_layout()

    output_path = DATA_ROOT / "audio_scatter_plots.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved scatter plots to {output_path}")


def main():
    print("=" * 70)
    print("MODEL COMPARISON & FUSION ANALYSIS")
    print("=" * 70)

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Download data
    download_data()

    # Load fold assignments and labels
    with open(FOLD_ASSIGNMENTS_FILE) as f:
        fold_assignments = json.load(f)
    with open(LABEL_FILE) as f:
        all_labels = json.load(f)

    # Get validation keys
    val_keys = set()
    for fold_id in range(4):
        val_keys.update(fold_assignments.get(f"fold_{fold_id}", []))
    print(f"\nValidation samples: {len(val_keys)}")

    # Download MERT embeddings
    download_mert_embeddings(val_keys)

    # Load audio models
    print("\nLoading audio models...")
    audio_models = load_audio_models(device)
    print(f"Loaded {len(audio_models)} audio model folds")

    # Generate audio predictions
    print("\nGenerating audio predictions...")
    audio_predictions = {}
    audio_labels = {}

    for fold_id in range(4):
        if fold_id not in audio_models:
            continue
        model = audio_models[fold_id]
        fold_keys = fold_assignments.get(f"fold_{fold_id}", [])

        for key in fold_keys:
            pred = predict_audio(model, key, device)
            if pred is not None and key in all_labels:
                audio_predictions[key] = pred
                audio_labels[key] = np.array(all_labels[key][:19])

    print(f"Generated {len(audio_predictions)} predictions")

    # Compute audio R2
    sorted_keys = sorted(audio_predictions.keys())
    preds_arr = np.array([audio_predictions[k] for k in sorted_keys])
    labels_arr = np.array([audio_labels[k] for k in sorted_keys])

    audio_overall_r2 = r2_score(labels_arr, preds_arr)
    audio_dim_r2 = {}
    for i, dim in enumerate(DIMENSIONS):
        audio_dim_r2[dim] = r2_score(labels_arr[:, i], preds_arr[:, i])

    print("\n" + "=" * 60)
    print("AUDIO MODEL RESULTS")
    print("=" * 60)
    print(f"\nOverall R2: {audio_overall_r2:.4f}")
    print("\nPer-dimension R2:")
    for dim, r2 in sorted(audio_dim_r2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dim:<25} {r2:+.4f}")

    # Sanity checks
    avg_dispersion = run_sanity_checks(audio_predictions, audio_labels, sorted_keys)

    # Comparison
    audio_wins = run_comparison(audio_dim_r2, audio_overall_r2)

    # Save scatter plots
    save_scatter_plots(audio_predictions, audio_labels, audio_dim_r2, sorted_keys)

    # Fusion testing (if symbolic predictions available)
    fusion_results = None
    if SYMBOLIC_PREDS_FILE.exists():
        with open(SYMBOLIC_PREDS_FILE) as f:
            symbolic_predictions = json.load(f)

        common_keys = sorted(
            set(audio_predictions.keys()) & set(symbolic_predictions.keys())
        )
        print(f"\nAligned samples for fusion: {len(common_keys)}")

        if len(common_keys) > 100:
            fusion_results = run_fusion(
                audio_predictions, symbolic_predictions, audio_labels, common_keys
            )
    else:
        print("\nSymbolic predictions not found - skipping fusion testing")
        print("Run the Thunder Compute notebook first to generate them")

    # Save results
    results = {
        "audio": {
            "overall_r2": float(audio_overall_r2),
            "per_dimension": {k: float(v) for k, v in audio_dim_r2.items()},
            "dispersion_ratio": float(avg_dispersion),
            "n_samples": len(audio_predictions),
        },
        "symbolic": {
            "overall_r2": SYMBOLIC_OVERALL_R2,
            "per_dimension": SYMBOLIC_DIM_R2,
        },
        "comparison": {
            "audio_wins": audio_wins,
            "symbolic_wins": 19 - audio_wins,
        },
    }

    if fusion_results:
        results["fusion"] = fusion_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Audio R2:     {audio_overall_r2:.4f}
    Symbolic R2:  {SYMBOLIC_OVERALL_R2:.4f}
    Audio wins:   {audio_wins}/19 dimensions
    """)

    if fusion_results:
        print(f"    Fusion R2:    {fusion_results['simple_fusion_r2']:.4f}")
        print(f"    Oracle R2:    {fusion_results['oracle_r2']:.4f}")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
