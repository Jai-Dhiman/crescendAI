#!/usr/bin/env python3
"""
My Piano Performance Analysis - Dataset Analysis
Understanding the PercePiano dataset for my recreation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# 19 perceptual dimensions from PercePiano paper
PERCEPTUAL_DIMENSIONS = [
    "Timing_Stable_Unstable",
    "Articulation_Short_Long", 
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark", 
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing"
]

def load_perceptual_labels() -> Dict:
    """Load PercePiano perceptual ratings"""
    # Try both relative paths depending on where script is run from
    possible_paths = [
        Path('../data/label_2round_mean_reg_19_with0_rm_highstd0.json'),  # from src/
        Path('data/label_2round_mean_reg_19_with0_rm_highstd0.json')     # from root
    ]
    
    labels_path = None
    for path in possible_paths:
        if path.exists():
            labels_path = path
            break
    
    if not labels_path:
        print("❌ Labels not found in expected locations:")
        for path in possible_paths:
            print(f"   Tried: {path}")
        print("Please copy the label file to my_implementation/data/")
        return {}
    
    with open(labels_path, 'r') as f:
        return json.load(f)

def analyze_dataset_characteristics(labels_data: Dict) -> Dict:
    """Analyze key dataset characteristics"""
    print("📊 DATASET CHARACTERISTICS")
    print("=" * 50)
    
    analysis = {}
    
    # Basic stats
    num_performances = len(labels_data)
    analysis['num_performances'] = num_performances
    print(f"Total performances: {num_performances}")
    
    # Extract ratings matrix
    ratings_matrix = []
    player_ids = []
    
    for performance, ratings in labels_data.items():
        perceptual_ratings = ratings[:-1]  # Last element is player_id
        player_id = ratings[-1]
        
        if len(perceptual_ratings) == len(PERCEPTUAL_DIMENSIONS):
            ratings_matrix.append(perceptual_ratings)
            player_ids.append(player_id)
    
    ratings_matrix = np.array(ratings_matrix)
    analysis['ratings_shape'] = ratings_matrix.shape
    
    # Rating statistics
    analysis['rating_stats'] = {
        'mean': float(np.mean(ratings_matrix)),
        'std': float(np.std(ratings_matrix)),
        'min': float(np.min(ratings_matrix)),
        'max': float(np.max(ratings_matrix))
    }
    
    print(f"Ratings matrix shape: {ratings_matrix.shape}")
    print(f"Rating range: [{analysis['rating_stats']['min']:.3f}, {analysis['rating_stats']['max']:.3f}]")
    print(f"Overall mean: {analysis['rating_stats']['mean']:.3f}")
    
    # Player distribution
    unique_players = np.unique(player_ids)
    analysis['num_players'] = len(unique_players)
    print(f"Unique players: {len(unique_players)}")
    
    # Dimension analysis
    print(f"\nPERCEPTUAL DIMENSIONS:")
    dimension_stats = {}
    
    for i, dim in enumerate(PERCEPTUAL_DIMENSIONS):
        dim_ratings = ratings_matrix[:, i]
        stats = {
            'mean': float(np.mean(dim_ratings)),
            'std': float(np.std(dim_ratings)),
            'min': float(np.min(dim_ratings)),
            'max': float(np.max(dim_ratings))
        }
        dimension_stats[dim] = stats
        
        print(f"  {dim[:30]:30} | Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f}")
    
    analysis['dimension_stats'] = dimension_stats
    
    return analysis

def find_extreme_performances(labels_data: Dict, n_extreme: int = 5) -> Dict:
    """Find performances with extreme ratings"""
    print(f"\n🎯 EXTREME PERFORMANCES (Top/Bottom {n_extreme})")
    print("=" * 50)
    
    # Calculate overall scores
    performance_scores = []
    for performance, ratings in labels_data.items():
        perceptual_ratings = ratings[:-1]
        if len(perceptual_ratings) == len(PERCEPTUAL_DIMENSIONS):
            overall_score = np.mean(perceptual_ratings)
            performance_scores.append({
                'name': performance,
                'score': overall_score,
                'player_id': ratings[-1],
                'ratings': perceptual_ratings
            })
    
    # Sort by score
    performance_scores.sort(key=lambda x: x['score'])
    
    # Get extremes
    lowest = performance_scores[:n_extreme]
    highest = performance_scores[-n_extreme:]
    
    print("LOWEST RATED:")
    for i, perf in enumerate(lowest):
        print(f"  {i+1}. {perf['name']}: {perf['score']:.3f} (Player {perf['player_id']})")
    
    print("\nHIGHEST RATED:")
    for i, perf in enumerate(highest):
        print(f"  {i+1}. {perf['name']}: {perf['score']:.3f} (Player {perf['player_id']})")
    
    return {
        'lowest': lowest,
        'highest': highest,
        'all_scores': performance_scores
    }

def analyze_dimension_correlations(labels_data: Dict) -> Dict:
    """Analyze correlations between perceptual dimensions"""
    print(f"\n🔗 DIMENSION CORRELATIONS")
    print("=" * 50)
    
    # Build ratings matrix
    ratings_list = []
    for performance, ratings in labels_data.items():
        perceptual_ratings = ratings[:-1]
        if len(perceptual_ratings) == len(PERCEPTUAL_DIMENSIONS):
            ratings_list.append(perceptual_ratings)
    
    ratings_df = pd.DataFrame(ratings_list, columns=PERCEPTUAL_DIMENSIONS)
    correlation_matrix = ratings_df.corr()
    
    # Find strong correlations (|r| > 0.5)
    strong_correlations = []
    for i in range(len(PERCEPTUAL_DIMENSIONS)):
        for j in range(i+1, len(PERCEPTUAL_DIMENSIONS)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_correlations.append({
                    'dim1': PERCEPTUAL_DIMENSIONS[i],
                    'dim2': PERCEPTUAL_DIMENSIONS[j], 
                    'correlation': float(corr_val)
                })
    
    # Sort by absolute correlation
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print("Strong correlations (|r| > 0.5):")
    for corr in strong_correlations[:10]:  # Top 10
        sign = "+" if corr['correlation'] > 0 else "-" 
        print(f"  {sign} {corr['dim1'][:25]:25} ↔ {corr['dim2'][:25]:25} | r={corr['correlation']:+.3f}")
    
    return {
        'correlation_matrix': correlation_matrix.values.tolist(),
        'strong_correlations': strong_correlations
    }

def main():
    """Main dataset analysis"""
    print("=== My Piano Performance Analysis - Dataset Analysis ===\n")
    
    # Load data
    labels_data = load_perceptual_labels()
    if not labels_data:
        return
    
    # Run analyses
    characteristics = analyze_dataset_characteristics(labels_data)
    extremes = find_extreme_performances(labels_data)
    correlations = analyze_dimension_correlations(labels_data)
    
    # Compile results
    full_analysis = {
        'dataset_characteristics': characteristics,
        'extreme_performances': extremes,
        'dimension_correlations': correlations,
        'perceptual_dimensions': PERCEPTUAL_DIMENSIONS
    }
    
    # Save analysis
    output_path = Path('../results/dataset_analysis.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to {output_path}")
    print(f"\n🎯 READY FOR NEXT PHASE: Building ML models to predict these {len(PERCEPTUAL_DIMENSIONS)} dimensions!")

if __name__ == "__main__":
    main()