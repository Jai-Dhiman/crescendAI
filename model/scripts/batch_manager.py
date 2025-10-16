#!/usr/bin/env python3
"""
Batch Manager for MAESTRO Dataset Processing
Helps manage incremental processing with disk space constraints.

With 123GB available and 108GB zip file, this enables:
1. Download full zip once (fits in available space)
2. Process files in batches using streaming_process_maestro.py
3. Track progress and disk usage
4. Delete zip after processing if needed to free 108GB

Usage:
    # Show current status
    python3 scripts/batch_manager.py --status
    
    # Start/resume processing (5 files for testing)
    python3 scripts/batch_manager.py --process --limit 5
    
    # Process larger batch
    python3 scripts/batch_manager.py --process --limit 50
    
    # Check if we should cleanup (delete zip after processing)
    python3 scripts/batch_manager.py --cleanup-check
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

def get_disk_usage(path: Path = Path('data')) -> Dict:
    """Get disk usage statistics."""
    stat = shutil.disk_usage(path)
    return {
        'total_gb': stat.total / (1024**3),
        'used_gb': stat.used / (1024**3),
        'free_gb': stat.free / (1024**3),
        'percent_used': (stat.used / stat.total) * 100
    }

def get_data_dir_sizes(data_dir: Path = Path('data')) -> Dict:
    """Get sizes of key data directories."""
    sizes = {}
    
    # Check zip file
    zip_path = data_dir / 'maestro-v3.0.0.zip'
    if zip_path.exists():
        sizes['zip_gb'] = zip_path.stat().st_size / (1024**3)
    else:
        sizes['zip_gb'] = 0
    
    # Check segments directory
    segments_dir = data_dir / 'segments'
    if segments_dir.exists():
        result = subprocess.run(
            ['du', '-sh', str(segments_dir)],
            capture_output=True,
            text=True
        )
        size_str = result.stdout.split()[0]
        sizes['segments_size'] = size_str
    else:
        sizes['segments_size'] = '0B'
    
    # Count segments
    if segments_dir.exists():
        segment_files = list(segments_dir.glob('*.wav'))
        sizes['segment_count'] = len(segment_files)
    else:
        sizes['segment_count'] = 0
    
    return sizes

def get_processing_progress(progress_file: Path = Path('data/.maestro_progress.json')) -> Optional[Dict]:
    """Load processing progress from file."""
    if not progress_file.exists():
        return None
    
    with open(progress_file, 'r') as f:
        return json.load(f)

def get_labeling_progress(
    labeled_file: Path = Path('data/manifests/segments_labeled.jsonl')
) -> int:
    """Count labeled segments."""
    if not labeled_file.exists():
        return 0
    
    with open(labeled_file, 'r') as f:
        return sum(1 for _ in f)

def print_status():
    """Print comprehensive status report."""
    print("=" * 70)
    print("MAESTRO BATCH PROCESSING STATUS")
    print("=" * 70)
    
    # Disk usage
    disk = get_disk_usage()
    print(f"\nDisk Space:")
    print(f"  Total: {disk['total_gb']:.1f} GB")
    print(f"  Used:  {disk['used_gb']:.1f} GB ({disk['percent_used']:.1f}%)")
    print(f"  Free:  {disk['free_gb']:.1f} GB")
    
    # Data directory sizes
    sizes = get_data_dir_sizes()
    print(f"\nData Directory:")
    print(f"  MAESTRO zip: {sizes['zip_gb']:.1f} GB")
    print(f"  Segments: {sizes['segments_size']} ({sizes['segment_count']} files)")
    
    # Processing progress
    progress = get_processing_progress()
    if progress:
        print(f"\nProcessing Progress:")
        print(f"  Files processed: {len(progress['files_processed'])}/1276")
        print(f"  Files failed: {len(progress['files_failed'])}")
        print(f"  Segments created: {progress['segments_created']}")
        if progress['processing_started']:
            print(f"  Started: {progress['processing_started']}")
        if progress['processing_completed']:
            print(f"  Completed: {progress['processing_completed']}")
    else:
        print(f"\nProcessing Progress: Not started")
    
    # Labeling progress
    labeled_count = get_labeling_progress()
    print(f"\nLabeling Progress:")
    print(f"  Labeled segments: {labeled_count}")
    if sizes['segment_count'] > 0:
        percent = (labeled_count / sizes['segment_count']) * 100
        print(f"  Completion: {percent:.1f}%")
    
    # Recommendations
    print(f"\nRecommendations:")
    if sizes['zip_gb'] == 0:
        print("  1. Download MAESTRO zip file:")
        print("     curl -C - -o data/maestro-v3.0.0.zip \\")
        print("       https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip")
    elif progress is None:
        print("  1. Start processing with test batch:")
        print("     python3 scripts/batch_manager.py --process --limit 5")
    elif sizes['segment_count'] == 0:
        print("  1. Continue processing:")
        print("     python3 scripts/batch_manager.py --process --limit 50")
    elif labeled_count == 0:
        print("  1. Start labeling segments:")
        print("     streamlit run labeling/quick_labeler.py")
    elif labeled_count < 1000:
        target = 1000 - labeled_count
        print(f"  1. Continue labeling (target: {target} more segments to reach 1K)")
        print("     streamlit run labeling/quick_labeler.py")
        if sizes['segment_count'] < 2000 and progress and len(progress['files_processed']) < 200:
            print(f"  2. Process more files for variety:")
            print("     python3 scripts/batch_manager.py --process --limit 50")
    elif labeled_count < 2000:
        target = 2000 - labeled_count
        print(f"  1. Continue labeling (target: {target} more segments to reach 2K)")
    else:
        print("  1. Excellent! You've reached 2K labeled segments!")
        print("  2. Consider training the bootstrap model:")
        print("     python3 train.py --data_dir ./data --experiment_name evaluator_v1")
    
    print("=" * 70)

def run_processing(limit: Optional[int] = None):
    """Run streaming processing."""
    cmd = ['python3', 'scripts/streaming_process_maestro.py']
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    print(f"Starting processing{f' (limit: {limit} files)' if limit else ''}...")
    print("This may take a while. Press Ctrl+C to stop (progress will be saved).")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted. Progress has been saved.")
        print("Run the same command again to resume.")
    except subprocess.CalledProcessError as e:
        print(f"\nProcessing failed with error code {e.returncode}")
        sys.exit(1)

def check_cleanup():
    """Check if zip file can be safely deleted to free space."""
    zip_path = Path('data/maestro-v3.0.0.zip')
    if not zip_path.exists():
        print("No zip file to cleanup.")
        return
    
    zip_size_gb = zip_path.stat().st_size / (1024**3)
    progress = get_processing_progress()
    
    print(f"MAESTRO zip file: {zip_size_gb:.1f} GB")
    
    if progress and len(progress['files_processed']) >= 1276:
        print(f"\nAll files have been processed!")
        print(f"You can safely delete the zip file to free {zip_size_gb:.1f} GB:")
        print(f"  rm data/maestro-v3.0.0.zip")
        print(f"\nNote: You can always re-download it later if needed.")
    elif progress:
        files_processed = len(progress['files_processed'])
        print(f"\nProcessing is incomplete: {files_processed}/1276 files processed")
        print(f"Keep the zip file to process remaining files.")
    else:
        print(f"\nProcessing hasn't started yet.")
        print(f"Keep the zip file for processing.")

def main():
    parser = argparse.ArgumentParser(
        description="Batch manager for MAESTRO processing with disk space management"
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help="Show current status"
    )
    parser.add_argument(
        '--process', '-p',
        action='store_true',
        help="Start/resume processing"
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help="Limit number of files to process"
    )
    parser.add_argument(
        '--cleanup-check', '-c',
        action='store_true',
        help="Check if zip can be safely deleted"
    )
    
    args = parser.parse_args()
    
    if args.cleanup_check:
        check_cleanup()
    elif args.process:
        run_processing(args.limit)
    else:
        # Default to showing status
        print_status()

if __name__ == '__main__':
    main()
