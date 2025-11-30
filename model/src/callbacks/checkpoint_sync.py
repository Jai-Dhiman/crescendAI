"""
Periodic checkpoint sync callback for Google Drive backup during training.

Ensures checkpoints are backed up regularly, protecting against crashes/OOM.
"""

import subprocess
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional
from datetime import datetime


class PeriodicCheckpointSync(pl.Callback):
    """
    Syncs checkpoints to Google Drive periodically during training.

    This provides safety for long training runs - if the process crashes,
    you won't lose more than `sync_every_n_steps` worth of progress.

    Uses rclone for efficient incremental syncing.
    """

    def __init__(
        self,
        local_checkpoint_dir: str,
        remote_path: str,
        sync_every_n_steps: int = 1000,
        sync_on_epoch_end: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize periodic checkpoint sync.

        Args:
            local_checkpoint_dir: Local directory containing checkpoints
            remote_path: rclone remote path (e.g., 'gdrive:checkpoints/model_name')
            sync_every_n_steps: Sync every N training steps (default: 1000)
            sync_on_epoch_end: Also sync at end of each epoch (default: True)
            verbose: Print sync status messages (default: True)
        """
        super().__init__()
        self.local_dir = Path(local_checkpoint_dir)
        self.remote_path = remote_path
        self.sync_every_n_steps = sync_every_n_steps
        self.sync_on_epoch_end = sync_on_epoch_end
        self.verbose = verbose
        self._last_sync_step = 0
        self._rclone_available = None

    def _check_rclone(self) -> bool:
        """Check if rclone is available and configured."""
        if self._rclone_available is not None:
            return self._rclone_available

        try:
            result = subprocess.run(
                ['rclone', 'listremotes'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._rclone_available = 'gdrive:' in result.stdout
            if not self._rclone_available and self.verbose:
                print("WARNING: rclone 'gdrive' remote not configured - checkpoint sync disabled")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._rclone_available = False
            if self.verbose:
                print("WARNING: rclone not available - checkpoint sync disabled")

        return self._rclone_available

    def _sync(self, context: str = "periodic"):
        """Perform the actual sync operation."""
        if not self._check_rclone():
            return

        if not self.local_dir.exists():
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        if self.verbose:
            print(f"[{timestamp}] Syncing checkpoints to GDrive ({context})...")

        try:
            result = subprocess.run(
                ['rclone', 'copy', str(self.local_dir), self.remote_path, '--quiet'],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )
            if result.returncode == 0:
                if self.verbose:
                    print(f"[{timestamp}] Sync complete")
            else:
                print(f"[{timestamp}] Sync failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"[{timestamp}] Sync timed out (continuing training)")
        except Exception as e:
            print(f"[{timestamp}] Sync error: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Sync periodically during training."""
        global_step = trainer.global_step

        # Check if we should sync
        if global_step > 0 and global_step - self._last_sync_step >= self.sync_every_n_steps:
            self._sync(context=f"step {global_step}")
            self._last_sync_step = global_step

    def on_train_epoch_end(self, trainer, pl_module):
        """Sync at end of each epoch."""
        if self.sync_on_epoch_end:
            self._sync(context=f"epoch {trainer.current_epoch} end")

    def on_validation_end(self, trainer, pl_module):
        """Sync after validation (when best checkpoint might be saved)."""
        self._sync(context="post-validation")


def sync_checkpoints_now(local_dir: str, remote_path: str, verbose: bool = True) -> bool:
    """
    Manually sync checkpoints to Google Drive.

    Utility function for use in notebooks when you want to force a sync.

    Args:
        local_dir: Local checkpoint directory
        remote_path: rclone remote path

    Returns:
        True if sync succeeded, False otherwise
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Check rclone availability
    try:
        result = subprocess.run(
            ['rclone', 'listremotes'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if 'gdrive:' not in result.stdout:
            print(f"[{timestamp}] ERROR: rclone 'gdrive' remote not configured")
            return False
    except Exception as e:
        print(f"[{timestamp}] ERROR: rclone not available: {e}")
        return False

    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"[{timestamp}] ERROR: Local directory does not exist: {local_dir}")
        return False

    if verbose:
        print(f"[{timestamp}] Syncing {local_dir} -> {remote_path}")

    try:
        result = subprocess.run(
            ['rclone', 'copy', str(local_path), remote_path, '--progress'],
            timeout=300,  # 5 minute timeout for manual sync
        )
        if result.returncode == 0:
            if verbose:
                print(f"[{timestamp}] Sync complete!")
            return True
        else:
            print(f"[{timestamp}] Sync failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[{timestamp}] Sync timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"[{timestamp}] Sync error: {e}")
        return False
