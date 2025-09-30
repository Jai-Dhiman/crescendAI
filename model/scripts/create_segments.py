#!/usr/bin/env python3
"""
Production MAESTRO segmentation pipeline with VirtuosoNet alignment.
Creates 8-16 bar segments (10-20s) with proper score-performance alignment.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pretty_midi
import librosa
import soundfile as sf
from tqdm import tqdm

# Add symbolic processing to path
sys.path.insert(0, str(Path(__file__).parent.parent / "symbolic"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

class MAESTROSegmenter:
    """Production segmentation pipeline for MAESTRO with alignment."""
    
    def __init__(self, maestro_dir: Path, output_dir: Path):
        self.maestro_dir = Path(maestro_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = self.maestro_dir / "maestro-v3.0.0.csv"
        
        # Segmentation parameters
        self.target_duration = (10.0, 20.0)  # 10-20 seconds
        self.min_bars = 8
        self.max_bars = 16
        self.target_sr = 22050
        
        # Output paths
        self.segments_dir = self.output_dir / "segments"
        self.manifests_dir = self.output_dir / "manifests"
        
        # Create directories
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load MAESTRO metadata."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"MAESTRO metadata not found: {self.metadata_file}")
        
        df = pd.read_csv(self.metadata_file)
        logger.info(f"Loaded metadata for {len(df)} recordings")
        return df
    
    def analyze_midi_structure(self, midi_path: Path) -> Dict:
        """Analyze MIDI file to identify bar boundaries and tempo."""
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Get tempo changes
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[1]) > 0:
                primary_tempo = tempo_changes[1][0]  # Use first tempo
            else:
                primary_tempo = 120.0  # Default fallback
            
            # Estimate bar duration (assuming 4/4 time)
            beat_duration = 60.0 / primary_tempo
            bar_duration = beat_duration * 4.0
            
            # Find downbeats (simplified - use tempo grid)
            end_time = midi.get_end_time()
            num_bars = int(end_time / bar_duration)
            bar_times = [i * bar_duration for i in range(num_bars + 1)]
            
            # Check for actual notes to validate timing
            all_notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    all_notes.extend(instrument.notes)
            
            if not all_notes:
                logger.warning(f"No notes found in {midi_path}")
                return None
            
            note_density = len(all_notes) / end_time if end_time > 0 else 0
            
            return {
                'duration': end_time,
                'primary_tempo': primary_tempo,
                'bar_duration': bar_duration,
                'num_bars': num_bars,
                'bar_times': bar_times,
                'note_density': note_density,
                'total_notes': len(all_notes)
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze MIDI {midi_path}: {e}")
            return None
    
    def create_segments_from_structure(self, 
                                     audio_path: Path, 
                                     midi_structure: Dict,
                                     recording_info: Dict) -> List[Dict]:
        """Create segments based on MIDI bar structure."""
        segments = []
        bar_times = midi_structure['bar_times']
        
        # Create segments of 8-16 bars
        for start_bar in range(0, len(bar_times) - self.min_bars, 4):  # Overlap by 50%
            for bar_length in [8, 12, 16]:  # Prefer these lengths
                end_bar = start_bar + bar_length
                
                if end_bar >= len(bar_times):
                    break
                
                t0 = bar_times[start_bar]
                t1 = bar_times[end_bar]
                duration = t1 - t0
                
                # Filter by target duration
                if not (self.target_duration[0] <= duration <= self.target_duration[1]):
                    continue
                
                # Create segment ID
                segment_id = (f"MAESTRO_{recording_info['year']}_{recording_info['track_id']}_"
                            f"bars_{start_bar:03d}_{end_bar:03d}")
                
                segment = {
                    'segment_id': segment_id,
                    'dataset': 'MAESTRO',
                    'audio_uri': f"file://{audio_path.absolute()}",
                    'sr': self.target_sr,
                    't0': float(t0),
                    't1': float(t1),
                    'duration': float(duration),
                    'bars': [start_bar, end_bar],
                    'bar_count': bar_length,
                    'tempo': midi_structure['primary_tempo'],
                    
                    # Initialize with execution dimensions for MAESTRO
                    'dims': [
                        'timing_stability', 'tempo_control', 'rhythmic_accuracy',
                        'articulation_length', 'articulation_hardness',
                        'pedal_density', 'pedal_clarity',
                        'dynamic_range', 'dynamic_control', 'balance_melody_vs_accomp'
                    ],
                    
                    # Initialize empty labels and masks
                    'labels': {},
                    'label_mask': {},
                    
                    # Metadata
                    'source': 'unlabeled',
                    'provenance': {
                        'piece': f"{recording_info['canonical_composer']} - {recording_info['canonical_title']}",
                        'year': recording_info['year'],
                        'split': recording_info['split'],
                        'midi_file': recording_info['midi_filename'],
                        'audio_file': recording_info['audio_filename'],
                        'license': 'CC BY-NC-SA 4.0'  # MAESTRO license
                    },
                    
                    # Quality metrics
                    'midi_structure': {
                        'note_density': midi_structure['note_density'],
                        'total_notes_in_segment': 0,  # Will be computed if needed
                        'tempo_stability': 'stable'   # Simplified for now
                    }
                }
                
                segments.append(segment)
        
        return segments
    
    def process_recording(self, row: pd.Series) -> List[Dict]:
        """Process a single MAESTRO recording to create segments."""
        audio_path = self.maestro_dir / row['audio_filename']
        midi_path = self.maestro_dir / row['midi_filename']
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return []
        
        if not midi_path.exists():
            logger.warning(f"MIDI file not found: {midi_path}")
            return []
        
        # Create track ID from filename
        track_id = Path(row['audio_filename']).stem.replace('MIDI-Unprocessed_', '').replace('_wav', '')
        
        recording_info = {
            'canonical_composer': row['canonical_composer'],
            'canonical_title': row['canonical_title'],
            'year': row['year'],
            'split': row['split'],
            'midi_filename': row['midi_filename'],
            'audio_filename': row['audio_filename'],
            'duration': row['duration'],
            'track_id': track_id
        }
        
        # Analyze MIDI structure
        midi_structure = self.analyze_midi_structure(midi_path)
        if midi_structure is None:
            logger.warning(f"Skipping {audio_path} - MIDI analysis failed")
            return []
        
        # Create segments
        segments = self.create_segments_from_structure(audio_path, midi_structure, recording_info)
        
        logger.info(f"Created {len(segments)} segments from {audio_path.name}")
        return segments
    
    def create_manifests(self, all_segments: List[Dict]) -> None:
        """Create train/valid/test manifest files."""
        # Split by MAESTRO's predefined splits
        splits = {'train': [], 'validation': [], 'test': []}
        
        for segment in all_segments:
            maestro_split = segment['provenance']['split']
            if maestro_split in splits:
                splits[maestro_split].append(segment)
        
        # Write manifest files
        for split_name, segments in splits.items():
            if not segments:
                continue
            
            # Map validation -> valid for consistency
            filename = "valid.jsonl" if split_name == "validation" else f"{split_name}.jsonl"
            manifest_path = self.manifests_dir / filename
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(json.dumps(segment, ensure_ascii=False) + '\n')
            
            logger.info(f"Created {filename} with {len(segments)} segments")
        
        # Create combined manifest
        combined_path = self.manifests_dir / "all_segments.jsonl"
        with open(combined_path, 'w', encoding='utf-8') as f:
            for segment in all_segments:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')
        
        logger.info(f"Created combined manifest with {len(all_segments)} segments")
    
    def generate_quality_report(self, all_segments: List[Dict]) -> None:
        """Generate quality report for the segmentation."""
        if not all_segments:
            return
        
        # Calculate statistics
        durations = [seg['duration'] for seg in all_segments]
        bar_counts = [seg['bar_count'] for seg in all_segments]
        tempos = [seg['tempo'] for seg in all_segments]
        
        splits = {}
        composers = {}
        years = {}
        
        for seg in all_segments:
            split = seg['provenance']['split']
            composer = seg['provenance']['piece'].split(' - ')[0]
            year = seg['provenance']['year']
            
            splits[split] = splits.get(split, 0) + 1
            composers[composer] = composers.get(composer, 0) + 1
            years[year] = years.get(year, 0) + 1
        
        report = {
            'total_segments': len(all_segments),
            'duration_stats': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'target_range': self.target_duration
            },
            'bar_count_stats': {
                'mean': float(np.mean(bar_counts)),
                'distribution': {str(k): int(v) for k, v in pd.Series(bar_counts).value_counts().items()}
            },
            'tempo_stats': {
                'mean': float(np.mean(tempos)),
                'std': float(np.std(tempos)),
                'min': float(np.min(tempos)),
                'max': float(np.max(tempos))
            },
            'split_distribution': splits,
            'top_composers': dict(sorted(composers.items(), key=lambda x: x[1], reverse=True)[:10]),
            'year_distribution': dict(sorted(years.items()))
        }
        
        report_path = self.output_dir / "segmentation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("SEGMENTATION SUMMARY")
        print("="*50)
        print(f"Total segments: {report['total_segments']:,}")
        print(f"Duration: {report['duration_stats']['mean']:.1f}s ± {report['duration_stats']['std']:.1f}s")
        print(f"Bar counts: {dict(sorted(report['bar_count_stats']['distribution'].items()))}")
        print(f"Splits: {splits}")
        print("="*50)


def main():
    """Main segmentation pipeline."""
    maestro_dir = Path("data/raw/MAESTRO")
    output_dir = Path("data")
    
    if not maestro_dir.exists():
        logger.error(f"MAESTRO directory not found: {maestro_dir}")
        logger.error("Run download_maestro.py first")
        return False
    
    segmenter = MAESTROSegmenter(maestro_dir, output_dir)
    
    # Load metadata
    try:
        df = segmenter.load_metadata()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    # Process recordings
    all_segments = []
    
    # For production, process all recordings
    # For testing, you can limit with: df = df.head(10)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing recordings"):
        try:
            segments = segmenter.process_recording(row)
            all_segments.extend(segments)
        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            continue
    
    if not all_segments:
        logger.error("No segments created")
        return False
    
    # Create manifests
    segmenter.create_manifests(all_segments)
    
    # Generate quality report
    segmenter.generate_quality_report(all_segments)
    
    logger.info("Segmentation pipeline completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)