"""GDrive sync utilities for experiment checkpoints and results."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set


def run_rclone(cmd: List[str], silent: bool = True) -> subprocess.CompletedProcess:
    """Run an rclone command."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result


def list_gdrive_experiments(gdrive_results_path: str) -> Set[str]:
    """List experiments that have results JSON in GDrive.

    Args:
        gdrive_results_path: GDrive path like 'gdrive:crescendai_data/checkpoints/audio_phase2'

    Returns:
        Set of experiment IDs that have .json result files
    """
    result = run_rclone(['rclone', 'lsf', gdrive_results_path])
    if result.returncode != 0:
        return set()

    experiments = set()
    for line in result.stdout.strip().split('\n'):
        if line.endswith('.json') and not line.startswith('phase2_'):
            exp_id = line[:-5]  # Remove .json
            experiments.add(exp_id)
    return experiments


def check_gdrive_checkpoints(gdrive_results_path: str, exp_id: str) -> bool:
    """Check if all 4 fold checkpoints exist in GDrive for an experiment.

    Args:
        gdrive_results_path: GDrive path like 'gdrive:crescendai_data/checkpoints/audio_phase2'
        exp_id: Experiment ID like 'B0_baseline'

    Returns:
        True if all 4 fold checkpoints exist
    """
    ckpt_path = f"{gdrive_results_path}/checkpoints/{exp_id}"
    result = run_rclone(['rclone', 'lsf', ckpt_path])
    if result.returncode != 0:
        return False

    files = set(result.stdout.strip().split('\n'))
    required = {f'fold{i}_best.ckpt' for i in range(4)}
    return required.issubset(files)


def get_completed_experiments(gdrive_results_path: str) -> Dict[str, float]:
    """Get all completed experiments from GDrive with their R2 scores.

    Args:
        gdrive_results_path: GDrive path like 'gdrive:crescendai_data/checkpoints/audio_phase2'

    Returns:
        Dict mapping experiment_id -> avg_r2 score
    """
    experiments = list_gdrive_experiments(gdrive_results_path)
    completed = {}

    for exp_id in experiments:
        if check_gdrive_checkpoints(gdrive_results_path, exp_id):
            # Fetch the results to get R2
            result = run_rclone([
                'rclone', 'cat', f'{gdrive_results_path}/{exp_id}.json'
            ])
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    r2 = data.get('summary', {}).get('avg_r2', None)
                    if r2 is not None:
                        completed[exp_id] = r2
                except json.JSONDecodeError:
                    pass

    return completed


def restore_experiment_from_gdrive(
    gdrive_results_path: str,
    exp_id: str,
    local_results_dir: Path,
    local_checkpoint_root: Path,
) -> bool:
    """Restore a single experiment's results and checkpoints from GDrive.

    Args:
        gdrive_results_path: GDrive path
        exp_id: Experiment ID
        local_results_dir: Local path for results JSON files
        local_checkpoint_root: Local root path for checkpoints

    Returns:
        True if restoration successful
    """
    local_results_dir = Path(local_results_dir)
    local_checkpoint_root = Path(local_checkpoint_root)

    # Check if exists in GDrive
    if not check_gdrive_checkpoints(gdrive_results_path, exp_id):
        return False

    # Create directories
    local_results_dir.mkdir(parents=True, exist_ok=True)
    exp_ckpt_dir = local_checkpoint_root / exp_id
    exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Download results JSON
    result = run_rclone([
        'rclone', 'copyto',
        f'{gdrive_results_path}/{exp_id}.json',
        str(local_results_dir / f'{exp_id}.json'),
    ])
    if result.returncode != 0:
        return False

    # Download checkpoints
    result = run_rclone([
        'rclone', 'copy',
        f'{gdrive_results_path}/checkpoints/{exp_id}',
        str(exp_ckpt_dir),
    ])

    return result.returncode == 0


def restore_all_from_gdrive(
    gdrive_results_path: str,
    local_results_dir: Path,
    local_checkpoint_root: Path,
    all_results_dict: Optional[Dict] = None,
) -> Dict[str, float]:
    """Restore all completed experiments from GDrive.

    Args:
        gdrive_results_path: GDrive path
        local_results_dir: Local path for results JSON files
        local_checkpoint_root: Local root path for checkpoints
        all_results_dict: Optional dict to populate with restored results

    Returns:
        Dict of restored experiment_id -> avg_r2
    """
    completed = get_completed_experiments(gdrive_results_path)
    restored = {}

    print(f"Found {len(completed)} completed experiments in GDrive")

    for exp_id, r2 in completed.items():
        print(f"  Restoring {exp_id} (R2={r2:.4f})...", end=' ')
        if restore_experiment_from_gdrive(
            gdrive_results_path, exp_id, local_results_dir, local_checkpoint_root
        ):
            restored[exp_id] = r2
            print("OK")

            # Load full results into provided dict
            if all_results_dict is not None:
                results_file = Path(local_results_dir) / f'{exp_id}.json'
                if results_file.exists():
                    with open(results_file) as f:
                        all_results_dict[exp_id] = json.load(f)
        else:
            print("FAILED")

    return restored


def sync_experiment_to_gdrive(
    exp_id: str,
    exp_results: Dict,
    local_results_dir: Path,
    local_checkpoint_root: Path,
    gdrive_results_path: str,
    all_results_dict: Optional[Dict] = None,
) -> bool:
    """Sync a single experiment's results and checkpoints to GDrive.

    Call this after each experiment completes.

    Args:
        exp_id: Experiment ID
        exp_results: Results dict from the experiment
        local_results_dir: Local path for results JSON files
        local_checkpoint_root: Local root path for checkpoints
        gdrive_results_path: GDrive path
        all_results_dict: Optional dict with all results (for aggregate JSON)

    Returns:
        True if sync successful
    """
    local_results_dir = Path(local_results_dir)
    local_checkpoint_root = Path(local_checkpoint_root)

    print(f"\nSyncing {exp_id} to GDrive...", end=' ')

    # Upload results JSON
    result_file = local_results_dir / f'{exp_id}.json'
    if result_file.exists():
        result = run_rclone([
            'rclone', 'copyto',
            str(result_file),
            f'{gdrive_results_path}/{exp_id}.json',
        ])
        if result.returncode != 0:
            print("FAILED (results)")
            return False

    # Upload checkpoints
    ckpt_dir = local_checkpoint_root / exp_id
    if ckpt_dir.exists():
        result = run_rclone([
            'rclone', 'copy',
            str(ckpt_dir),
            f'{gdrive_results_path}/checkpoints/{exp_id}',
        ])
        if result.returncode != 0:
            print("FAILED (checkpoints)")
            return False

    # Update aggregate results JSON
    if all_results_dict is not None:
        aggregate_file = local_results_dir / 'phase2_all_results.json'
        with open(aggregate_file, 'w') as f:
            json.dump(all_results_dict, f, indent=2)

        run_rclone([
            'rclone', 'copyto',
            str(aggregate_file),
            f'{gdrive_results_path}/phase2_all_results.json',
        ])

    r2 = exp_results.get('summary', {}).get('avg_r2', 0)
    print(f"OK (R2={r2:.4f})")
    return True


def should_run_experiment(
    exp_id: str,
    checkpoint_root: Path,
    results_dir: Path,
    gdrive_results_path: str,
    completed_cache: Optional[Dict[str, float]] = None,
) -> bool:
    """Check if experiment needs to run (not already complete).

    Checks both local files and GDrive to determine if experiment is complete.
    Use this BEFORE extracting embeddings to avoid unnecessary work.

    Args:
        exp_id: Experiment ID like 'B0_baseline'
        checkpoint_root: Local root path for checkpoints
        results_dir: Local path for results JSON files
        gdrive_results_path: GDrive path like 'gdrive:crescendai_data/checkpoints/audio_phase2'
        completed_cache: Optional pre-fetched dict of completed experiments (avoids repeated GDrive calls)

    Returns:
        True if experiment should run, False if already complete
    """
    from .runner import experiment_completed, load_existing_results

    # Check local first (fast)
    if experiment_completed(exp_id, checkpoint_root) and load_existing_results(exp_id, results_dir):
        print(f"SKIP {exp_id}: already complete locally")
        return False

    # Check GDrive (use cache if provided)
    if completed_cache is not None:
        if exp_id in completed_cache:
            print(f"SKIP {exp_id}: already complete on GDrive (R2={completed_cache[exp_id]:.4f})")
            return False
    else:
        completed = get_completed_experiments(gdrive_results_path)
        if exp_id in completed:
            print(f"SKIP {exp_id}: already complete on GDrive (R2={completed[exp_id]:.4f})")
            return False

    return True


def print_experiment_status(
    all_experiment_ids: List[str],
    completed_experiments: Dict[str, float],
) -> None:
    """Print a status table of experiments.

    Args:
        all_experiment_ids: List of all experiment IDs in order
        completed_experiments: Dict of completed experiment_id -> r2
    """
    print("\n" + "="*60)
    print("EXPERIMENT STATUS")
    print("="*60)
    print(f"{'Experiment':<25} {'Status':<12} {'R2':>10}")
    print("-"*60)

    for exp_id in all_experiment_ids:
        if exp_id in completed_experiments:
            status = "DONE"
            r2_str = f"{completed_experiments[exp_id]:.4f}"
        else:
            status = "PENDING"
            r2_str = "---"
        print(f"{exp_id:<25} {status:<12} {r2_str:>10}")

    done = len(completed_experiments)
    total = len(all_experiment_ids)
    remaining = total - done
    print("-"*60)
    print(f"Completed: {done}/{total} | Remaining: {remaining}")
    print("="*60 + "\n")
