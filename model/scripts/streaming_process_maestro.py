#!/usr/bin/env python3
"""
Streaming MAESTRO processor for constrained disk space.
Extracts, processes, and deletes files one-by-one from the zip archive.

This approach keeps disk usage minimal:
- Download zip once (~114GB)
- Extract single audio + MIDI file (~100-500MB)
- Process into segments (~10-50MB)
- Delete extracted files immediately
- Repeat for all files

Total peak disk usage: ~115GB (zip + single file + segments)
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track processing progress across runs."""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.state = self.load()
    
    def load(self) -> Dict:
        """Load progress state from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'download_complete': False,
            'files_processed': [],
            'files_failed': [],
            'segments_created': 0,
            'last_updated': None,
            'processing_started': None,
            'processing_completed': None
        }
    
    def save(self):
        """Save progress state to file atomically."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Write to temp file first, then atomic rename
        temp_file = self.progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.progress_file)
    
    def mark_file_processed(self, filename: str, segments_count: int):
        """Mark a file as successfully processed."""
        if filename not in self.state['files_processed']:
            self.state['files_processed'].append(filename)
        self.state['segments_created'] += segments_count
        
        # Remove from failed list if present
        if filename in self.state['files_failed']:
            self.state['files_failed'].remove(filename)
        
        self.save()
    
    def mark_file_failed(self, filename: str, error: str):
        """Mark a file as failed."""
        if filename not in self.state['files_failed']:
            self.state['files_failed'].append(filename)
        self.save()
    
    def is_processed(self, filename: str) -> bool:
        """Check if file has been processed."""
        return filename in self.state['files_processed']


class StreamingMAESTROProcessor:
    """Stream-process MAESTRO dataset from zip file."""
    
    def __init__(self, 
                 zip_path: Path,
                 output_dir: Path,
                 progress_file: Path,
                 target_sr: int = 22050,
                 min_duration: float = 10.0,
                 max_duration: float = 20.0):
        self.zip_path = Path(zip_path)
        self.output_dir = Path(output_dir)
        self.progress_file = progress_file
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Create output directories
        self.segments_dir = self.output_dir / "segments"
        self.manifests_dir = self.output_dir / "manifests"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker
        self.progress = ProgressTracker(progress_file)
        
        # Temp directory for extraction
        self.temp_dir = None
    
    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """Check if sufficient disk space is available."""
        stat = shutil.disk_usage(self.output_dir)
        available_gb = stat.free / (1024**3)
        
        if available_gb < required_gb:
            logger.error(f"Insufficient disk space. Need {required_gb}GB, have {available_gb:.1f}GB")
            return False
        
        logger.info(f"Disk space check passed: {available_gb:.1f}GB available")
        return True
    
    def list_audio_files(self) -> List[Tuple[str, str]]:
        """List all audio/MIDI file pairs in the zip."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        
        pairs = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            # First, find the CSV metadata file
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV metadata file found in zip")
            
            csv_file = csv_files[0]
            logger.info(f"Found metadata file: {csv_file}")
            
            # Extract and read CSV
            with zf.open(csv_file) as f:
                df = pd.read_csv(f)
            
            logger.info(f"Found {len(df)} recordings in metadata")
            
            # Build list of audio/MIDI pairs
            for _, row in df.iterrows():
                audio_file = row['audio_filename']
                midi_file = row['midi_filename']
                
                # Find full paths in zip
                audio_path = None
                midi_path = None
                
                for name in zf.namelist():
                    if name.endswith(audio_file):
                        audio_path = name
                    if name.endswith(midi_file):
                        midi_path = name
                
                if audio_path and midi_path:
                    pairs.append((audio_path, midi_path, row.to_dict()))
                else:
                    logger.warning(f"Missing files for: {audio_file}")
        
        return pairs
    
    def analyze_midi_structure(self, midi_data: bytes) -> Optional[Dict]:
        """Analyze MIDI structure from bytes."""
        try:
            # Write MIDI to temp file for pretty_midi
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                f.write(midi_data)
                temp_midi_path = f.name
            
            try:
                midi = pretty_midi.PrettyMIDI(temp_midi_path)
                
                # Get tempo
                tempo_changes = midi.get_tempo_changes()
                primary_tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
                
                # Calculate bar boundaries (assuming 4/4)
                beat_duration = 60.0 / primary_tempo
                bar_duration = beat_duration * 4.0
                
                end_time = midi.get_end_time()
                num_bars = int(end_time / bar_duration)
                bar_times = [i * bar_duration for i in range(num_bars + 1)]
                
                # Get notes
                all_notes = []
                for instrument in midi.instruments:
                    if not instrument.is_drum:
                        all_notes.extend(instrument.notes)
                
                if not all_notes:
                    return None
                
                return {
                    'duration': end_time,
                    'primary_tempo': primary_tempo,
                    'bar_duration': bar_duration,
                    'num_bars': num_bars,
                    'bar_times': bar_times,
                    'note_density': len(all_notes) / end_time if end_time > 0 else 0,
                    'total_notes': len(all_notes)
                }
            finally:
                Path(temp_midi_path).unlink()
        
        except Exception as e:
            logger.warning(f"Failed to analyze MIDI: {e}")
            return None
    
    def create_segments(self, 
                       audio_data: bytes,
                       midi_structure: Dict,
                       metadata: Dict) -> List[Dict]:
        """Create segments from audio bytes."""
        segments = []
        
        try:
            # Load audio from bytes
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_data)
                temp_audio_path = f.name
            
            try:
                y, sr = librosa.load(temp_audio_path, sr=self.target_sr, mono=True)
                
                # Create segments based on bar structure
                bar_times = midi_structure['bar_times']
                
                # Try different bar lengths
                for start_bar in range(0, len(bar_times) - 8, 4):
                    for bar_length in [8, 12, 16]:
                        end_bar = start_bar + bar_length
                        
                        if end_bar >= len(bar_times):
                            continue
                        
                        t0 = bar_times[start_bar]
                        t1 = bar_times[end_bar]
                        duration = t1 - t0
                        
                        # Filter by duration
                        if not (self.min_duration <= duration <= self.max_duration):
                            continue
                        
                        # Extract audio segment
                        start_sample = int(t0 * self.target_sr)
                        end_sample = int(t1 * self.target_sr)
                        
                        if end_sample > len(y):
                            continue
                        
                        segment_audio = y[start_sample:end_sample]
                        
                        # Generate segment ID
                        year = metadata.get('year', 'unknown')
                        composer = metadata.get('canonical_composer', 'unknown').replace(' ', '_')
                        title = metadata.get('canonical_title', 'unknown')[:30].replace(' ', '_')
                        
                        segment_id = f"MAESTRO_{year}_{composer}_{title}_bars_{start_bar:03d}_{end_bar:03d}"
                        segment_id = "".join(c for c in segment_id if c.isalnum() or c in ['_', '-'])
                        
                        # Save segment audio
                        segment_path = self.segments_dir / f"{segment_id}.wav"
                        sf.write(segment_path, segment_audio, self.target_sr)
                        
                        # Create manifest entry
                        segment = {
                            'segment_id': segment_id,
                            'dataset': 'MAESTRO',
                            'audio_uri': f"file://{segment_path.absolute()}",
                            'sr': self.target_sr,
                            't0': float(t0),
                            't1': float(t1),
                            'duration': float(duration),
                            'bars': [start_bar, end_bar],
                            'bar_count': bar_length,
                            'tempo': midi_structure['primary_tempo'],
                            'dims': [
                                'timing_stability', 'tempo_control', 'rhythmic_accuracy',
                                'articulation_length', 'articulation_hardness',
                                'pedal_density', 'pedal_clarity',
                                'dynamic_range', 'dynamic_control', 'balance_melody_vs_accomp'
                            ],
                            'labels': {},
                            'label_mask': {},
                            'source': 'unlabeled',
                            'provenance': {
                                'piece': f"{metadata.get('canonical_composer', 'unknown')} - {metadata.get('canonical_title', 'unknown')}",
                                'year': year,
                                'split': metadata.get('split', 'unknown'),
                                'license': 'CC BY-NC-SA 4.0'
                            }
                        }
                        
                        segments.append(segment)
            
            finally:
                Path(temp_audio_path).unlink()
        
        except Exception as e:
            logger.error(f"Failed to create segments: {e}")
        
        return segments
    
    def process_file_pair(self, 
                          audio_path: str, 
                          midi_path: str, 
                          metadata: Dict,
                          zf: zipfile.ZipFile) -> int:
        """Process a single audio/MIDI pair from zip."""
        try:
            # Extract MIDI and analyze structure
            logger.info(f"Analyzing: {Path(midi_path).name}")
            midi_data = zf.read(midi_path)
            midi_structure = self.analyze_midi_structure(midi_data)
            
            if not midi_structure:
                logger.warning(f"Failed to analyze MIDI structure")
                return 0
            
            # Extract audio
            logger.info(f"Processing: {Path(audio_path).name}")
            audio_data = zf.read(audio_path)
            
            # Create segments
            segments = self.create_segments(audio_data, midi_structure, metadata)
            
            if segments:
                # Write segments to manifest
                manifest_file = self.manifests_dir / "segments_unlabeled.jsonl"
                with open(manifest_file, 'a') as f:
                    for seg in segments:
                        f.write(json.dumps(seg) + '\n')
                
                logger.info(f"Created {len(segments)} segments")
                return len(segments)
            else:
                logger.warning(f"No valid segments created")
                return 0
        
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            raise
    
    def process_all(self, limit: Optional[int] = None) -> Dict:
        """Process all files in the dataset."""
        if not self.check_disk_space(10.0):
            raise RuntimeError("Insufficient disk space")
        
        # Mark start time
        if not self.progress.state['processing_started']:
            self.progress.state['processing_started'] = datetime.now().isoformat()
            self.progress.save()
        
        # List all file pairs
        logger.info("Scanning zip file for audio/MIDI pairs...")
        pairs = self.list_audio_files()
        logger.info(f"Found {len(pairs)} file pairs")
        
        if limit:
            pairs = pairs[:limit]
            logger.info(f"Processing limited to {limit} files")
        
        # Filter out already processed
        pairs_to_process = [
            (audio, midi, meta) for audio, midi, meta in pairs
            if not self.progress.is_processed(audio)
        ]
        
        logger.info(f"{len(pairs_to_process)} files remaining to process")
        
        if not pairs_to_process:
            logger.info("All files already processed!")
            return self.get_statistics()
        
        # Process each pair
        total_segments = 0
        failed_count = 0
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for audio_path, midi_path, metadata in tqdm(pairs_to_process, desc="Processing files"):
                try:
                    segments_count = self.process_file_pair(audio_path, midi_path, metadata, zf)
                    
                    if segments_count > 0:
                        self.progress.mark_file_processed(audio_path, segments_count)
                        total_segments += segments_count
                    else:
                        self.progress.mark_file_failed(audio_path, "No segments created")
                        failed_count += 1
                    
                    # Periodic disk space check
                    if total_segments % 10 == 0:
                        if not self.check_disk_space(5.0):
                            logger.error("Running low on disk space, stopping")
                            break
                
                except Exception as e:
                    logger.error(f"Failed to process {audio_path}: {e}")
                    self.progress.mark_file_failed(audio_path, str(e))
                    failed_count += 1
        
        # Mark completion
        self.progress.state['processing_completed'] = datetime.now().isoformat()
        self.progress.save()
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'files_processed': len(self.progress.state['files_processed']),
            'files_failed': len(self.progress.state['files_failed']),
            'segments_created': self.progress.state['segments_created'],
            'processing_started': self.progress.state['processing_started'],
            'processing_completed': self.progress.state['processing_completed']
        }


def main():
    parser = argparse.ArgumentParser(
        description="Stream-process MAESTRO dataset with minimal disk usage"
    )
    parser.add_argument(
        '--zip', '-z',
        type=Path,
        default=Path('data/maestro-v3.0.0.zip'),
        help="Path to MAESTRO zip file"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data'),
        help="Output directory for segments and manifests"
    )
    parser.add_argument(
        '--progress', '-p',
        type=Path,
        default=Path('data/.maestro_progress.json'),
        help="Progress tracking file"
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help="Show current progress status"
    )
    
    args = parser.parse_args()
    
    # Show status if requested
    if args.status:
        if args.progress.exists():
            tracker = ProgressTracker(args.progress)
            print("=" * 60)
            print("MAESTRO PROCESSING STATUS")
            print("=" * 60)
            print(f"Files processed: {len(tracker.state['files_processed'])}")
            print(f"Files failed: {len(tracker.state['files_failed'])}")
            print(f"Segments created: {tracker.state['segments_created']}")
            print(f"Started: {tracker.state['processing_started']}")
            print(f"Completed: {tracker.state['processing_completed']}")
            print("=" * 60)
        else:
            print("No progress file found - processing not yet started")
        return
    
    # Verify zip file exists
    if not args.zip.exists():
        logger.error(f"Zip file not found: {args.zip}")
        logger.info("Download MAESTRO first:")
        logger.info(f"  wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip -P {args.zip.parent}")
        sys.exit(1)
    
    # Create processor and run
    processor = StreamingMAESTROProcessor(
        zip_path=args.zip,
        output_dir=args.output,
        progress_file=args.progress
    )
    
    logger.info("Starting MAESTRO stream processing...")
    logger.info(f"Zip file: {args.zip}")
    logger.info(f"Output: {args.output}")
    
    stats = processor.process_all(limit=args.limit)
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
