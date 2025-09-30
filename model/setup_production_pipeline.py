#!/usr/bin/env python3
"""
Production pipeline setup for CrescendAI Evaluator model.
Orchestrates MAESTRO download, segmentation, and labeling environment setup.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionPipelineSetup:
    """Orchestrates the complete production pipeline setup."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else Path.cwd()
        self.scripts_dir = self.model_dir / "scripts"
        self.data_dir = self.model_dir / "data"

    def check_environment(self) -> bool:
        """Check if the environment is properly set up."""
        try:
            # Check if we're in a virtual environment
            result = subprocess.run(
                [sys.executable, "-c", "import sys; print(sys.prefix)"],
                capture_output=True,
                text=True,
            )
            venv_path = result.stdout.strip()

            if ".venv" not in venv_path:
                logger.warning(
                    "Not in a virtual environment. Run 'source .venv/bin/activate' first."
                )
                return False

            # Check required imports
            required_packages = [
                "torch",
                "pytorch_lightning",
                "librosa",
                "soundfile",
                "pandas",
                "numpy",
                "pretty_midi",
                "streamlit",
                "requests",
            ]

            for package in required_packages:
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", f"import {package}"], capture_output=True
                    )
                    if result.returncode != 0:
                        logger.error(f"Package {package} not installed")
                        return False
                except Exception:
                    logger.error(f"Failed to check package {package}")
                    return False

            logger.info("Environment check passed")
            return True

        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            return False

    def run_script(self, script_name: str, args: List[str] = None) -> bool:
        """Run a pipeline script with error handling."""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=self.model_dir, check=True)
            logger.info(f"✅ {script_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {script_name} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"❌ {script_name} failed: {e}")
            return False

    def setup_directories(self) -> bool:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir / "raw" / "MAESTRO",
            self.data_dir / "manifests",
            self.data_dir / "splits",
            self.data_dir / "anchors",
            self.scripts_dir,
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            logger.info("Directory structure verified")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def check_maestro_status(self) -> str:
        """Check MAESTRO dataset status."""
        maestro_dir = self.data_dir / "raw" / "MAESTRO"
        metadata_file = maestro_dir / "maestro-v3.0.0.csv"

        if metadata_file.exists():
            return "extracted"

        zip_file = self.data_dir / "maestro-v3.0.0.zip"
        if zip_file.exists():
            return "downloaded"

        return "missing"

    def print_status_report(self) -> None:
        """Print current pipeline status."""
        print("\n" + "=" * 60)
        print("CRESCENDAI EVALUATOR - PRODUCTION PIPELINE STATUS")
        print("=" * 60)

        # Environment
        env_ok = self.check_environment()
        print(f"Environment: {'✅ Ready' if env_ok else '❌ Issues detected'}")

        # Directories
        dirs_ok = all(
            [
                (self.data_dir / "raw" / "MAESTRO").exists(),
                (self.data_dir / "manifests").exists(),
                (self.data_dir / "splits").exists(),
                (self.data_dir / "anchors").exists(),
            ]
        )
        print(f"Directories: {'✅ Ready' if dirs_ok else '❌ Missing'}")

        # MAESTRO status
        maestro_status = self.check_maestro_status()
        status_symbols = {
            "extracted": "✅ Extracted and ready",
            "downloaded": "⚠️  Downloaded but not extracted",
            "missing": "❌ Not downloaded",
        }
        print(f"MAESTRO Dataset: {status_symbols[maestro_status]}")

        # Manifests
        train_manifest = self.data_dir / "manifests" / "all_segments.jsonl"
        manifests_ok = train_manifest.exists() and train_manifest.stat().st_size > 0
        print(f"Segment Manifests: {'✅ Ready' if manifests_ok else '❌ Missing'}")

        # Anchors
        anchors_file = self.data_dir / "anchors" / "anchors.json"
        anchors_ok = anchors_file.exists()
        print(
            f"Anchor Definitions: {'✅ Complete (16 dimensions)' if anchors_ok else '❌ Missing'}"
        )

        # Labeling interface
        labeler_file = self.model_dir / "labeling" / "quick_labeler.py"
        labeler_ok = labeler_file.exists()
        print(f"Labeling Interface: {'✅ Ready' if labeler_ok else '❌ Missing'}")

        print("=" * 60)

        # Next steps
        print("\nNEXT STEPS:")
        if not env_ok:
            print("1. Fix environment issues (run 'uv sync --extra labeling')")
        elif maestro_status == "missing":
            print("1. Download MAESTRO dataset (run download step)")
        elif maestro_status == "downloaded":
            print("1. Extract MAESTRO dataset")
        elif not manifests_ok:
            print("1. Generate segments and manifests")
        else:
            print("1. ✅ Ready to begin labeling!")
            print("   Run: streamlit run labeling/quick_labeler.py")

    def run_full_pipeline(self, skip_download: bool = False) -> bool:
        """Run the complete production pipeline."""
        logger.info("Starting production pipeline setup...")

        # Step 1: Environment check
        if not self.check_environment():
            logger.error("Environment check failed. Install dependencies first.")
            return False

        # Step 2: Setup directories
        if not self.setup_directories():
            return False

        # Step 3: Download MAESTRO (if needed)
        if not skip_download:
            maestro_status = self.check_maestro_status()
            if maestro_status in ["missing", "downloaded"]:
                logger.info("Downloading and extracting MAESTRO dataset...")
                if not self.run_script("download_maestro.py"):
                    return False

        # Step 4: Create segments
        maestro_status = self.check_maestro_status()
        if maestro_status == "extracted":
            logger.info("Creating segments from MAESTRO dataset...")
            if not self.run_script("create_segments.py"):
                return False
        else:
            logger.warning("MAESTRO not extracted, skipping segmentation")

        logger.info("Production pipeline setup completed!")
        self.print_status_report()
        return True


def main():
    """Main entry point with command line options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CrescendAI Evaluator Production Pipeline"
    )
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument(
        "--download-only", action="store_true", help="Only download MAESTRO"
    )
    parser.add_argument(
        "--segments-only",
        action="store_true",
        help="Only create segments (requires MAESTRO)",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip MAESTRO download"
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    pipeline = ProductionPipelineSetup()

    if args.status:
        pipeline.print_status_report()
        return

    if args.download_only:
        success = pipeline.run_script("download_maestro.py")
        exit(0 if success else 1)

    if args.segments_only:
        success = pipeline.run_script("create_segments.py")
        exit(0 if success else 1)

    if args.full or len(sys.argv) == 1:  # Default to full pipeline
        success = pipeline.run_full_pipeline(skip_download=args.skip_download)
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
