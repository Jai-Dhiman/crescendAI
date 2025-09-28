#!/usr/bin/env python3
"""
Script to download and manage datasets for the classical piano performance feedback system.
Currently supports MAESTRO dataset v3.0.0.
"""
import os
import sys
import json
import zipfile
import requests
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import csv
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Project paths
PROJECT_ROOT = Path("/app/classical_piano_performance_2225")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MAESTRO_RAW_DIR = RAW_DATA_DIR / "maestro"
MAESTRO_PROCESSED_DIR = PROCESSED_DATA_DIR / "maestro"
MANIFEST_FILE = MAESTRO_RAW_DIR / "manifest.json"
# MAESTRO dataset information
MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MAESTRO_ZIP_FILENAME = "maestro-v3.0.0.zip"
MAESTRO_METADATA_CSV = "maestro-v3.0.0.csv"
MAESTRO_METADATA_JSON = "maestro-v3.0.0.json"
def download_file(url: str, destination: Path) -> bool:
    """
    Download a file from URL to destination path with progress bar.
    Args:
        url: URL to download from
        destination: Path to save the file
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {destination}")
        # Create directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stream download to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    # Log progress every 10MB
                    if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                        progress = (downloaded_size / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        logger.info(f"Download completed: {destination}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False
def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract ZIP file to specified directory.
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        # Create extraction directory
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extraction completed")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False
def validate_maestro_dataset(extract_dir: Path) -> bool:
    """
    Validate that MAESTRO dataset was extracted correctly.
    Args:
        extract_dir: Directory where dataset was extracted
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        logger.info("Validating MAESTRO dataset")
        # Check if metadata files exist
        metadata_csv = extract_dir / MAESTRO_METADATA_CSV
        metadata_json = extract_dir / MAESTRO_METADATA_JSON
        if not metadata_csv.exists():
            logger.error(f"Metadata CSV file not found: {metadata_csv}")
            return False
        if not metadata_json.exists():
            logger.error(f"Metadata JSON file not found: {metadata_json}")
            return False
        # Check if year directories exist (should have at least one)
        year_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if len(year_dirs) == 0:
            logger.error("No year directories found in extracted dataset")
            return False
        logger.info(f"Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}")
        # Check if we have audio and MIDI files
        audio_files = list(extract_dir.rglob("*.wav"))
        midi_files = list(extract_dir.rglob("*.midi"))
        logger.info(f"Found {len(audio_files)} audio files and {len(midi_files)} MIDI files")
        if len(audio_files) == 0 or len(midi_files) == 0:
            logger.warning("No audio or MIDI files found - this might be normal for partial extraction")
        logger.info("MAESTRO dataset validation passed")
        return True
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False
def generate_maestro_manifest(extract_dir: Path) -> Dict:
    """
    Generate manifest file with dataset metadata.
    Args:
        extract_dir: Directory where dataset was extracted
    Returns:
        Dict: Manifest data
    """
    try:
        logger.info("Generating MAESTRO manifest")
        # Read metadata CSV
        metadata_csv = extract_dir / MAESTRO_METADATA_CSV
        manifest_data = {
            "dataset": "MAESTRO v3.0.0",
            "version": "3.0.0",
            "description": "MIDI and Audio Edited for Synchronous TRacks and Organization",
            "source": "https://magenta.tensorflow.org/datasets/maestro",
            "files": []
        }
        if metadata_csv.exists():
            files_data = []
            with open(metadata_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert to our manifest format
                    file_entry = {
                        "canonical_composer": row.get("canonical_composer", ""),
                        "canonical_title": row.get("canonical_title", ""),
                        "split": row.get("split", ""),
                        "year": row.get("year", ""),
                        "midi_filename": row.get("midi_filename", ""),
                        "audio_filename": row.get("audio_filename", ""),
                        "duration": float(row.get("duration", 0)) if row.get("duration") else 0
                    }
                    files_data.append(file_entry)
            manifest_data["files"] = files_data
            logger.info(f"Manifest generated with {len(files_data)} entries")
        else:
            logger.warning("Metadata CSV not found, creating empty manifest")
        return manifest_data
    except Exception as e:
        logger.error(f"Manifest generation failed: {e}")
        return {"error": str(e)}
def save_manifest(manifest_data: Dict, manifest_path: Path) -> bool:
    """
    Save manifest data to JSON file.
    Args:
        manifest_data: Manifest data to save
        manifest_path: Path to save manifest
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        logger.info(f"Saving manifest to {manifest_path}")
        # Create directory if it doesn't exist
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        logger.info("Manifest saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")
        return False
def download_maestro() -> bool:
    """
    Download and extract MAESTRO dataset.
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting MAESTRO dataset download")
        # Create raw data directory
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Download ZIP file
        zip_path = RAW_DATA_DIR / MAESTRO_ZIP_FILENAME
        if not zip_path.exists():
            if not download_file(MAESTRO_URL, zip_path):
                logger.error("Failed to download MAESTRO dataset")
                return False
        else:
            logger.info(f"MAESTRO ZIP already exists at {zip_path}")
        # Extract ZIP file
        if not extract_zip(zip_path, MAESTRO_RAW_DIR):
            logger.error("Failed to extract MAESTRO dataset")
            return False
        # Validate extraction
        if not validate_maestro_dataset(MAESTRO_RAW_DIR):
            logger.error("MAESTRO dataset validation failed")
            return False
        # Generate and save manifest
        manifest_data = generate_maestro_manifest(MAESTRO_RAW_DIR)
        if not save_manifest(manifest_data, MANIFEST_FILE):
            logger.error("Failed to save manifest")
            return False
        logger.info("MAESTRO dataset download and setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"MAESTRO download failed: {e}")
        return False
def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Download and manage datasets for piano performance feedback system")
    parser.add_argument("--download", action="store_true", help="Download MAESTRO dataset")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity")
    args = parser.parse_args()
    if args.download:
        if not download_maestro():
            logger.error("MAESTRO download failed")
            sys.exit(1)
        else:
            logger.info("MAESTRO download completed successfully")
    elif args.verify:
        if validate_maestro_dataset(MAESTRO_RAW_DIR):
            logger.info("MAESTRO dataset verification passed")
        else:
            logger.error("MAESTRO dataset verification failed")
            sys.exit(1)
    else:
        parser.print_help()
if __name__ == "__main__":
    main()
