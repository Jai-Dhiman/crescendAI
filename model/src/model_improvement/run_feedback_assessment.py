"""Run Experiment 4: MIDI-as-context feedback assessment.

Usage:
    cd model
    uv run python -m model_improvement.run_feedback_assessment

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from model_improvement.audio_encoders import MuQLoRAModel
from model_improvement.feedback_assessment import (
    build_condition_a_prompt,
    build_condition_b_prompt,
    build_judge_prompt,
    parse_judge_response,
)
from model_improvement.layer1_validation import score_competition_segments
from model_improvement.midi_comparison import structured_midi_comparison
from model_improvement.taxonomy import DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints/model_improvement"
PERCEPIANO_DIR = DATA_DIR / "percepiano_cache"
RESULTS_DIR = DATA_DIR / "experiment4_results"


def _load_anthropic_client():
    """Load Anthropic client. Raises if API key not set."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Call Claude API and return text response."""
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _select_segments(
    labels: dict,
    n_top: int = 5,
    n_middle: int = 10,
    n_bottom: int = 5,
    piece_prefix: str = "Schubert_D960_mv3",
) -> list[str]:
    """Select PercePiano segments spanning the quality range for one piece."""
    piece_keys = [k for k in labels if k.startswith(piece_prefix)]
    if not piece_keys:
        logger.warning("No segments found for %s, using all keys", piece_prefix)
        piece_keys = list(labels.keys())

    scored = [(k, np.mean(labels[k][:6])) for k in piece_keys]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    selected.extend([k for k, _ in scored[:n_top]])
    mid_start = len(scored) // 2 - n_middle // 2
    selected.extend([k for k, _ in scored[mid_start:mid_start + n_middle]])
    selected.extend([k for k, _ in scored[-n_bottom:]])

    # Deduplicate while preserving order
    seen = set()
    result = []
    for k in selected:
        if k not in seen:
            seen.add(k)
            result.append(k)

    return result[:n_top + n_middle + n_bottom]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load A1 model
    a1_ckpt = sorted(CHECKPOINT_DIR.glob("A1/fold_3/*.ckpt"))[0]
    logger.info("Loading A1 from %s", a1_ckpt.name)
    a1_model = MuQLoRAModel.load_from_checkpoint(str(a1_ckpt), use_pretrained_muq=False)
    a1_model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    a1_model = a1_model.to(device)

    # Load labels and embeddings
    from model_improvement.taxonomy import load_composite_labels
    labels = load_composite_labels(DATA_DIR / "composite_labels/composite_labels.json")

    # Select 20 segments
    selected = _select_segments(labels)
    logger.info("Selected %d segments for assessment", len(selected))

    # Load pre-extracted embeddings
    emb_cache = PERCEPIANO_DIR / "_muq_file_cache"
    embeddings = {}
    for key in selected:
        pt_path = emb_cache / f"{key}.pt"
        if pt_path.exists():
            embeddings[key] = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            logger.warning("Embedding not found for %s", key)

    # Score with A1
    segment_scores = score_competition_segments(a1_model, embeddings)

    # Build MIDI comparisons (requires Score MIDI)
    midi_comparisons = {}
    midi_dir = DATA_DIR / "percepiano_midi"
    if midi_dir.exists():
        import pretty_midi
        for key in selected:
            perf_midi_path = midi_dir / f"{key}.mid"
            score_key = key.rsplit("_", 1)[0] + "_Score"
            score_midi_path = midi_dir / f"{score_key}.mid"

            if perf_midi_path.exists() and score_midi_path.exists():
                perf_pm = pretty_midi.PrettyMIDI(str(perf_midi_path))
                score_pm = pretty_midi.PrettyMIDI(str(score_midi_path))

                perf_notes = [
                    {"pitch": n.pitch, "velocity": n.velocity,
                     "onset": n.start, "duration": n.end - n.start}
                    for inst in perf_pm.instruments for n in inst.notes
                ]
                score_notes = [
                    {"pitch": n.pitch, "velocity": n.velocity,
                     "onset": n.start, "duration": n.end - n.start}
                    for inst in score_pm.instruments for n in inst.notes
                ]

                midi_comparisons[key] = structured_midi_comparison(perf_notes, score_notes)
    else:
        logger.warning("No MIDI directory at %s -- Condition B will be limited", midi_dir)

    # Generate observations and judge
    client = _load_anthropic_client()
    results = []
    student_context = {"level": "intermediate", "session_count": 12}

    for key in selected:
        if key not in segment_scores:
            continue

        scores = {dim: float(segment_scores[key][d]) for d, dim in enumerate(DIMENSIONS)}

        # Condition A: scores only
        prompt_a = build_condition_a_prompt(scores, student_context)
        obs_a = _call_llm(client, prompt_a)

        # Condition B: scores + MIDI data
        midi_comp = midi_comparisons.get(key)
        if midi_comp:
            prompt_b = build_condition_b_prompt(scores, student_context, midi_comp)
        else:
            logger.warning("No MIDI data for %s, skipping", key)
            continue

        obs_b = _call_llm(client, prompt_b)

        # Judge
        judge_prompt = build_judge_prompt(obs_a, obs_b)
        judge_response = _call_llm(client, judge_prompt)

        try:
            judgment = parse_judge_response(judge_response)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse judge response for %s: %s", key, e)
            judgment = {"winner": "unknown", "error": str(e)}

        result = {
            "segment_id": key,
            "scores": scores,
            "observation_a": obs_a,
            "observation_b": obs_b,
            "judgment": judgment,
            "has_midi": midi_comp is not None,
        }
        results.append(result)
        logger.info(
            "  %s: winner=%s", key, judgment.get("winner", "unknown")
        )

    # Save results
    output_path = RESULTS_DIR / "feedback_assessment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    wins_b = sum(1 for r in results if r["judgment"].get("winner") == "B")
    wins_a = sum(1 for r in results if r["judgment"].get("winner") == "A")
    total = wins_a + wins_b
    if total > 0:
        win_rate_b = wins_b / total * 100
        gate = "BUILD" if win_rate_b > 65 else "SKIP" if win_rate_b < 55 else "BORDERLINE"
        logger.info("\n=== Experiment 4 Results ===")
        logger.info("Condition B (MIDI context) wins: %d/%d (%.0f%%)", wins_b, total, win_rate_b)
        logger.info("Condition A (scores only) wins: %d/%d (%.0f%%)", wins_a, total, 100 - win_rate_b)
        logger.info("Decision: %s score comparison pipeline", gate)
    else:
        logger.warning("No valid comparisons completed")

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
