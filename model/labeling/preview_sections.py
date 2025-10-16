#!/usr/bin/env python3
"""
Preview Sections - Utility to analyze section-based organization

This script helps you understand how your segments will be organized
in the section-based labeler before you start labeling.

Usage:
    python3 labeling/preview_sections.py [path_to_manifest.jsonl]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_sections(segments: List[Dict]):
    """Analyze how segments are organized by piece and section"""
    
    # Group by (piece, bars)
    groups = defaultdict(list)
    piece_info = defaultdict(lambda: {"performers": set(), "sections": set()})
    
    for segment in segments:
        piece = segment["provenance"]["piece"]
        bars = tuple(segment["bars"])
        year = segment["provenance"]["year"]
        
        groups[(piece, bars)].append(segment)
        piece_info[piece]["performers"].add(year)
        piece_info[piece]["sections"].add(bars)
    
    print("=== MAESTRO Section-Based Organization ===\n")
    
    # Overall statistics
    total_segments = len(segments)
    total_sections = len(groups)
    total_pieces = len(piece_info)
    
    print(f"📊 **Overview:**")
    print(f"   • Total segments: {total_segments}")
    print(f"   • Total sections: {total_sections} (unique piece-bar combinations)")
    print(f"   • Total pieces: {total_pieces}")
    print()
    
    # Piece-by-piece breakdown
    print(f"🎼 **Pieces and Performers:**")
    for piece, info in sorted(piece_info.items()):
        performers = sorted(info["performers"])
        section_count = len(info["sections"])
        print(f"   • {piece}")
        print(f"     - Performers: {performers} ({len(performers)} total)")
        print(f"     - Sections: {section_count}")
        print()
    
    # Section detail analysis
    print(f"🎯 **Section Analysis:**")
    sections_by_performer_count = defaultdict(int)
    
    for (piece, bars), segments_list in groups.items():
        performer_count = len(segments_list)
        sections_by_performer_count[performer_count] += 1
    
    print(f"   Sections by number of performers:")
    for performer_count in sorted(sections_by_performer_count.keys()):
        section_count = sections_by_performer_count[performer_count]
        print(f"   • {performer_count} performers: {section_count} sections")
    
    print()
    
    # Show some example sections
    print(f"📋 **Example Sections (first 10):**")
    sorted_groups = sorted(groups.items(), key=lambda x: (x[0][0], x[0][1][0]))
    
    for i, ((piece, bars), segments_list) in enumerate(sorted_groups[:10]):
        performers = [seg["provenance"]["year"] for seg in segments_list]
        print(f"   {i+1:2d}. {piece}")
        print(f"       Bars {bars[0]}-{bars[1]} | Performers: {sorted(performers)}")
    
    if len(sorted_groups) > 10:
        print(f"       ... and {len(sorted_groups) - 10} more sections")
    
    print()
    
    # Labeling workflow preview
    print(f"⚡ **Labeling Workflow Preview:**")
    print(f"   The section-based labeler will present sections in this order:")
    print(f"   1. Label ALL performers for: {sorted_groups[0][0][0]} bars {sorted_groups[0][0][1][0]}-{sorted_groups[0][0][1][1]}")
    if len(sorted_groups) > 1:
        print(f"   2. Label ALL performers for: {sorted_groups[1][0][0]} bars {sorted_groups[1][0][1][0]}-{sorted_groups[1][0][1][1]}")
    if len(sorted_groups) > 2:
        print(f"   3. Label ALL performers for: {sorted_groups[2][0][0]} bars {sorted_groups[2][0][1][0]}-{sorted_groups[2][0][1][1]}")
    print(f"   ... continuing through all {len(sorted_groups)} sections")
    
    print()
    print(f"✅ **Benefits of this approach:**")
    print(f"   • Consistent comparative evaluation across performers")
    print(f"   • Better understanding of performance variations in specific passages")
    print(f"   • More efficient for calibrating your scoring across performers")
    print(f"   • Easier to maintain consistency within the same musical context")


def main():
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    else:
        # Default path
        manifest_path = Path("data/manifests/segments_unlabeled.jsonl")
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        print("Usage: python3 labeling/preview_sections.py [path_to_manifest.jsonl]")
        return
    
    try:
        segments = load_jsonl(manifest_path)
        if not segments:
            print(f"❌ Manifest is empty: {manifest_path}")
            return
        
        analyze_sections(segments)
        
        print(f"💡 **Next Steps:**")
        print(f"   To start section-based labeling:")
        print(f"   streamlit run labeling/section_based_labeler.py")
        print()
        
    except Exception as e:
        print(f"❌ Error analyzing manifest: {e}")


if __name__ == "__main__":
    main()