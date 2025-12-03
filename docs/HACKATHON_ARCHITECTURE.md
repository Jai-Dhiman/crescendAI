# CrescendAI: Technical Architecture

**xAI Hackathon | December 6-7, 2025**

---

## System Overview

CrescendAI uses a **Grok-as-Orchestrator** architecture. The user talks to Grok via voice; Grok calls our MCP server to control a companion app that displays visualizations and plays audio.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GROK APP (Voice Mode)                                 │
│                                                                         │
│  User: "Start my piano lesson. Session code A7X2."                      │
│  User: "Analyze my performance"                                         │
│  User: "Why do I rush there?"                                           │
│                                                                         │
│  Grok responds with personality + calls MCP tools                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Remote MCP Tools
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CRESCENDAI MCP SERVER (Modal)                        │
│                                                                         │
│  Tools (return data + teaching_context for guided personality):         │
│  • start_lesson(session_code, piece) → confirms connection              │
│  • analyze_performance(session_code) → analysis + teaching hints        │
│  • play_reference(session_code, measures) → triggers companion          │
│  • play_comparison(session_code, measures) → A/B playback               │
│  • highlight_measures(session_code, measures, color)                    │
│  • set_practice_tempo(session_code, tempo_percent)                      │
│  • get_measure_detail(session_code, measure) → deep dive                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (real-time)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              COMPANION APP (Browser - Next.js)                          │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌──────────────────┐   │
│  │ Session Code    │  │ Piano Roll + Score   │  │ Recording Select │   │
│  │    A 7 X 2      │  │ ████░░████░░██████   │  │ ○ Amateur        │   │
│  │ ✅ Connected    │  │ [highlighted areas]  │  │ ● Professional   │   │
│  └─────────────────┘  └──────────────────────┘  └──────────────────┘   │
│                                                                         │
│  Pre-loaded recordings: amateur.mp3, professional.mp3                  │
│  Fallback MIDI: amateur.mid, professional.mid                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Session Linking

### How It Works

1. **Companion app generates 4-character code** on load (e.g., `A7X2`)
2. **User tells Grok the code** in natural speech
3. **Grok passes code to MCP** via `start_lesson(session_code="A7X2")`
4. **MCP server links the session** and notifies companion via WebSocket

### Implementation

```python
# mcp_server/sessions.py
import secrets
import string
from dataclasses import dataclass, field
from typing import Dict, Optional
import asyncio

@dataclass
class Session:
    code: str
    piece: Optional[str] = None
    selected_recording: Optional[str] = None  # "amateur" or "professional"
    websocket: Optional[any] = None
    analysis_result: Optional[dict] = None
    
# Active sessions indexed by code
sessions: Dict[str, Session] = {}

def generate_session_code() -> str:
    """Generate a memorable 4-character code (letters + numbers, no ambiguous chars)."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # No 0/O, 1/I/L
    return ''.join(secrets.choice(alphabet) for _ in range(4))

def create_session(websocket) -> str:
    """Called when companion app connects."""
    code = generate_session_code()
    sessions[code] = Session(code=code, websocket=websocket)
    return code

def get_session(code: str) -> Optional[Session]:
    """Retrieve session by code (case-insensitive)."""
    return sessions.get(code.upper())
```

### Companion App Session Display

```typescript
// companion/components/SessionCode.tsx
import { useEffect, useState } from 'react';
import { useWebSocket } from '../lib/websocket';

export function SessionCode() {
  const [code, setCode] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const { socket, status } = useWebSocket();

  useEffect(() => {
    if (socket && status === 'connected') {
      // Request session code from server
      socket.send(JSON.stringify({ action: 'get_session_code' }));
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'session_code') {
          setCode(data.code);
        }
        if (data.type === 'grok_connected') {
          setConnected(true);
        }
      };
    }
  }, [socket, status]);

  return (
    <div className="session-display">
      {code ? (
        <>
          <div className="code-label">Your session code:</div>
          <div className="code-digits">
            {code.split('').map((char, i) => (
              <span key={i} className="digit">{char}</span>
            ))}
          </div>
          <div className="instruction">
            Tell Grok: "My session code is {code}"
          </div>
          <div className={`status ${connected ? 'connected' : 'waiting'}`}>
            {connected ? '✅ Connected to Grok' : '⏳ Waiting for Grok...'}
          </div>
        </>
      ) : (
        <div className="loading">Connecting...</div>
      )}
    </div>
  );
}
```

---

## Data Flow

### Flow 1: Lesson Start

```
1. User opens companion app
   → Companion connects to MCP server via WebSocket
   → Server generates code "A7X2", sends to companion
   → Companion displays: "Your session code: A7X2"

2. User (to Grok): "Start my piano lesson. Session code A7X2. 
                    I'm working on the Pathétique."

3. Grok calls MCP tool:
   → start_lesson(session_code="A7X2", piece="pathetique")

4. MCP Server:
   → Finds session by code
   → Sets session.piece = "pathetique"
   → Sends WebSocket to companion: {action: "lesson_started", piece: "pathetique"}
   → Returns to Grok: {
       status: "connected",
       piece_title: "Pathétique Sonata, 3rd Movement", 
       composer: "Ludwig van Beethoven",
       message: "Connected to companion. The student can now select their recording.",
       teaching_context: {
         piece_background: "Beethoven's most dramatic sonata, written during his hearing loss",
         what_to_listen_for: "Rhythmic precision in the rondo theme, dynamic contrasts"
       }
     }

5. Companion updates:
   → Shows "✅ Connected to Grok"
   → Shows piece title
   → Enables recording selection (Amateur / Professional)

6. Grok (to user): "Got it—I'm connected and I've loaded the Pathétique, 
                    third movement. Select your recording in the companion 
                    app and tell me when you're ready for feedback."
```

### Flow 2: Performance Analysis

```
1. User selects "Amateur" recording in companion app
   → Companion sends WebSocket: {action: "recording_selected", recording: "amateur"}
   → Server updates session.selected_recording = "amateur"

2. User (to Grok): "Analyze my performance"

3. Grok calls MCP tool:
   → analyze_performance(session_code="A7X2")

4. MCP Server:
   → Gets session, confirms recording is selected
   → Loads audio file: data/pathetique_amateur.mp3
   → Transcribes with Basic Pitch (or loads fallback MIDI)
   → Aligns to reference score with DTW
   → Analyzes: timing, accuracy, dynamics
   → Returns structured analysis + teaching_context

5. Response to Grok:
   {
     "analysis": {
       "overall_accuracy": 0.84,
       "tempo_stability": 0.71,
       "dynamic_range": 0.65,
       "duration_seconds": 45.2,
       "measures_analyzed": 68,
       "measures_with_issues": [
         {"measure": 43, "issues": ["rushing"], "timing_deviation_ms": -89},
         {"measure": 47, "issues": ["rushing", "uneven"], "timing_deviation_ms": -112},
         {"measure": 51, "issues": ["rushing"], "timing_deviation_ms": -95}
       ],
       "strengths": [
         "Strong dynamic commitment in opening (measures 1-8)",
         "Clean articulation in the second theme"
       ]
     },
     "teaching_context": {
       "lead_with": "The dramatic opening chords show real commitment—this student understands the character",
       "primary_focus": "Consistent rushing in measures 43-51 (sixteenth-note passages)",
       "root_cause_hint": "Likely anxiety about the technical difficulty causing 'survival mode'",
       "analogy": "The left hand octaves should be like a heartbeat—steady and inevitable, grounding the drama",
       "practice_strategy": "Isolate measures 43-51 at 60% tempo, left hand alone first",
       "personality_note": "Be encouraging but specific. Use the heartbeat analogy."
     }
   }

6. Grok streams response, incorporating teaching_context naturally

7. Grok calls additional tools:
   → highlight_measures(session_code="A7X2", measures=[43,47,51], color="red")
   → play_comparison(session_code="A7X2", start_measure=43, end_measure=51)
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| User Interface | Grok App (voice mode) | Native voice, no custom UI |
| Companion App | Next.js 15 + React | Fast dev, Vercel deployment |
| MCP Server | FastAPI + Modal | MCP protocol, GPU access |
| Real-time Comm | WebSocket | Low latency |
| Audio Transcription | Spotify Basic Pitch | 94% F1, <3s on GPU |
| MIDI Analysis | pretty_midi + music21 | Industry standard |
| Score Alignment | librosa (DTW) | Fast, deterministic |
| Visualization | html-midi-player | Free piano roll |
| GPU Hosting | Modal | Memory snapshots |
| Companion Hosting | Vercel | Fast, free tier |

---

## Component 1: MCP Server

### Tool Definitions with Teaching Context

```python
# mcp_server/tools.py
from mcp import tool
from pydantic import BaseModel
from typing import List, Optional
import pretty_midi

class MeasureIssue(BaseModel):
    measure: int
    issues: List[str]
    timing_deviation_ms: float
    wrong_notes: Optional[List[int]] = None

class TeachingContext(BaseModel):
    lead_with: str
    primary_focus: str
    root_cause_hint: Optional[str] = None
    analogy: Optional[str] = None
    practice_strategy: str
    personality_note: str

class PerformanceAnalysis(BaseModel):
    overall_accuracy: float
    tempo_stability: float
    dynamic_range: float
    duration_seconds: float
    measures_analyzed: int
    measures_with_issues: List[MeasureIssue]
    strengths: List[str]

class AnalysisResponse(BaseModel):
    analysis: PerformanceAnalysis
    teaching_context: TeachingContext

@tool
async def start_lesson(session_code: str, piece: str) -> dict:
    """
    Connect to a companion app session and load a piece for the lesson.
    Call this when the user provides their session code and piece name.
    
    Args:
        session_code: The 4-character code displayed on the companion app
        piece: The piece to practice (e.g., "pathetique")
    """
    session = get_session(session_code)
    if not session:
        return {
            "status": "error",
            "message": f"Session {session_code} not found. Ask the user to check the code on their companion app."
        }
    
    session.piece = piece
    piece_info = PIECES.get(piece, PIECES["pathetique"])
    
    await session.websocket.send_json({
        "action": "lesson_started",
        "piece": piece,
        "piece_info": piece_info
    })
    
    return {
        "status": "connected",
        "piece_title": piece_info["title"],
        "composer": piece_info["composer"],
        "message": f"Connected to companion app. {piece_info['title']} is loaded. The student can now select their recording.",
        "teaching_context": {
            "piece_background": piece_info["background"],
            "what_to_listen_for": piece_info["listen_for"]
        }
    }

@tool
async def analyze_performance(session_code: str) -> AnalysisResponse:
    """
    Analyze the selected recording against the reference score.
    Call this when the user asks for feedback on their performance.
    
    Returns detailed analysis plus teaching hints to guide your response.
    Lead with encouragement, then address the primary focus area.
    """
    session = get_session(session_code)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    if not session.selected_recording:
        return {
            "status": "waiting",
            "message": "No recording selected yet. Ask the user to select their recording in the companion app."
        }
    
    # Load the audio file
    recording = RECORDINGS[session.piece][session.selected_recording]
    audio_path = recording["audio_path"]
    
    # Transcribe (with MIDI fallback)
    try:
        midi = await transcribe_audio(audio_path)
    except Exception as e:
        # Fallback to pre-transcribed MIDI
        midi = pretty_midi.PrettyMIDI(recording["midi_fallback_path"])
    
    # Load reference and analyze
    reference = load_reference(session.piece)
    alignment = align_to_score(midi, reference)
    analysis = analyze_performance_metrics(midi, reference, alignment)
    
    # Generate teaching context based on analysis
    teaching_context = generate_teaching_context(analysis, session.piece)
    
    # Store for later tools (highlight, comparison)
    session.analysis_result = {
        "student_midi": midi,
        "reference": reference,
        "alignment": alignment,
        "analysis": analysis
    }
    
    return AnalysisResponse(
        analysis=analysis,
        teaching_context=teaching_context
    )

@tool
async def highlight_measures(
    session_code: str,
    measures: List[int],
    color: str = "red"
) -> dict:
    """
    Highlight specific measures on the companion app's piano roll.
    Use this to draw attention to problem areas while explaining them.
    
    Colors: "red" (problems), "yellow" (attention), "green" (good)
    """
    session = get_session(session_code)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    await session.websocket.send_json({
        "action": "highlight_measures",
        "measures": measures,
        "color": color
    })
    
    return {
        "status": "highlighted",
        "measures": measures,
        "message": f"Measures {measures} are now highlighted in {color} on the student's screen."
    }

@tool
async def play_reference(
    session_code: str,
    start_measure: int,
    end_measure: int,
    tempo_percent: int = 100
) -> dict:
    """
    Play the reference (correct) version of specific measures.
    Use this to demonstrate how a passage should sound.
    """
    session = get_session(session_code)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    await session.websocket.send_json({
        "action": "play_reference",
        "start_measure": start_measure,
        "end_measure": end_measure,
        "tempo_percent": tempo_percent
    })
    
    return {
        "status": "playing",
        "message": f"Playing reference version of measures {start_measure}-{end_measure} at {tempo_percent}% tempo."
    }

@tool
async def play_comparison(
    session_code: str,
    start_measure: int,
    end_measure: int,
    student_first: bool = True
) -> dict:
    """
    Play the student's version and reference back-to-back for comparison.
    Highly effective for demonstrating timing and accuracy differences.
    """
    session = get_session(session_code)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    await session.websocket.send_json({
        "action": "play_comparison",
        "start_measure": start_measure,
        "end_measure": end_measure,
        "student_first": student_first
    })
    
    order = "student then reference" if student_first else "reference then student"
    return {
        "status": "playing_comparison",
        "message": f"Playing comparison ({order}) for measures {start_measure}-{end_measure}. The difference should be audible."
    }

@tool
async def set_practice_tempo(
    session_code: str,
    tempo_percent: int
) -> dict:
    """
    Set a practice tempo for the student to try.
    Use this when suggesting they slow down a difficult passage.
    """
    session = get_session(session_code)
    if not session:
        return {"status": "error", "message": "Session not found"}
    
    await session.websocket.send_json({
        "action": "set_tempo",
        "tempo_percent": tempo_percent
    })
    
    return {
        "status": "tempo_set",
        "message": f"Practice tempo set to {tempo_percent}%. The student can now play along with the slowed reference."
    }

@tool
async def get_measure_detail(
    session_code: str,
    measure: int
) -> dict:
    """
    Get detailed analysis of a specific measure.
    Use this when the student asks about a particular measure or wants to understand an issue deeply.
    """
    session = get_session(session_code)
    if not session or not session.analysis_result:
        return {"status": "error", "message": "No analysis available"}
    
    analysis = session.analysis_result["analysis"]
    measure_data = next(
        (m for m in analysis.measures_with_issues if m.measure == measure),
        None
    )
    
    if not measure_data:
        return {
            "measure": measure,
            "status": "clean",
            "message": f"Measure {measure} looks good! No significant issues detected.",
            "teaching_context": {
                "note": "Acknowledge this is a strong point before moving to areas that need work."
            }
        }
    
    return {
        "measure": measure,
        "issues": measure_data.issues,
        "timing_deviation_ms": measure_data.timing_deviation_ms,
        "wrong_notes": measure_data.wrong_notes,
        "teaching_context": {
            "explanation": get_issue_explanation(measure_data.issues),
            "fix_suggestion": get_fix_suggestion(measure_data.issues)
        }
    }
```

### Teaching Context Generator

```python
# mcp_server/teaching.py
from typing import List
from .analysis import PerformanceAnalysis, MeasureIssue

def generate_teaching_context(analysis: PerformanceAnalysis, piece: str) -> dict:
    """
    Generate pedagogically-sound teaching hints based on analysis.
    These guide Grok's response style and content.
    """
    
    # Find the dominant issue pattern
    all_issues = []
    for m in analysis.measures_with_issues:
        all_issues.extend(m.issues)
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    primary_issue = max(issue_counts, key=issue_counts.get) if issue_counts else None
    
    # Generate context based on primary issue
    context = {
        "lead_with": generate_encouragement(analysis),
        "primary_focus": generate_primary_focus(primary_issue, analysis),
        "personality_note": "Be encouraging but specific. The student wants real feedback, not empty praise."
    }
    
    # Add issue-specific guidance
    if primary_issue == "rushing":
        context["root_cause_hint"] = "Rushing usually indicates anxiety about technical difficulty—the brain tries to 'get through' hard passages"
        context["analogy"] = "The left hand should be like a heartbeat—steady and inevitable, grounding the drama above"
        context["practice_strategy"] = "Isolate the rushed passages at 50-60% tempo. Practice left hand alone until it's boring, then add right hand."
        
    elif primary_issue == "dragging":
        context["root_cause_hint"] = "Dragging often means the student is thinking too hard about each note instead of feeling the phrase"
        context["analogy"] = "Think of the phrase like a sentence—you don't pause between every word"
        context["practice_strategy"] = "Practice with a metronome, but set it to click only on beat 1 of each measure. Feel the flow between clicks."
        
    elif primary_issue == "wrong_notes":
        context["root_cause_hint"] = "Wrong notes in consistent spots usually mean the fingering needs revision"
        context["practice_strategy"] = "Go back to the score. Circle the wrong notes. Practice just those spots, slowly, with correct fingering."
        
    elif primary_issue == "flat_dynamics":
        context["root_cause_hint"] = "Flat dynamics often mean the student is focused on 'getting the notes' rather than making music"
        context["analogy"] = "Dynamics are like storytelling—imagine you're narrating the emotional arc"
        context["practice_strategy"] = "Exaggerate! Play the louds twice as loud and softs twice as soft. Then dial it back to taste."
    
    return context

def generate_encouragement(analysis: PerformanceAnalysis) -> str:
    """Generate genuine encouragement based on strengths."""
    if analysis.strengths:
        return analysis.strengths[0]
    
    if analysis.overall_accuracy > 0.9:
        return "Excellent note accuracy—the hard work on learning the notes is paying off"
    elif analysis.overall_accuracy > 0.8:
        return "Solid accuracy overall—you clearly know this piece"
    elif analysis.dynamic_range > 0.7:
        return "Good dynamic range—you're not just playing notes, you're making music"
    elif analysis.tempo_stability > 0.8:
        return "Your tempo is steady—that's harder than it sounds"
    else:
        return "You're tackling a challenging piece—that takes courage"

def generate_primary_focus(issue: str, analysis: PerformanceAnalysis) -> str:
    """Generate the primary teaching focus."""
    if not issue:
        return "Refinement and interpretation—the fundamentals are solid"
    
    # Find measures with this issue
    affected_measures = [m.measure for m in analysis.measures_with_issues if issue in m.issues]
    measure_str = ", ".join(str(m) for m in affected_measures[:5])
    
    if issue == "rushing":
        return f"Tempo control in measures {measure_str}—the sixteenth notes are running ahead of the beat"
    elif issue == "dragging":
        return f"Forward momentum in measures {measure_str}—the tempo is pulling back"
    elif issue == "wrong_notes":
        return f"Note accuracy in measures {measure_str}—a few pitches need attention"
    elif issue == "uneven":
        return f"Evenness in measures {measure_str}—some notes are getting swallowed"
    elif issue == "flat_dynamics":
        return f"Dynamic contrast—the piece needs more light and shade"
    
    return f"Technical work in measures {measure_str}"
```

---

## Component 2: Audio Transcription with MIDI Fallback

```python
# analysis/transcription.py
import pretty_midi
from basic_pitch.inference import predict
from pathlib import Path

# Pre-loaded fallback MIDI files
MIDI_FALLBACKS = {
    "pathetique": {
        "amateur": "data/pathetique_amateur.mid",
        "professional": "data/pathetique_professional.mid"
    }
}

async def transcribe_audio(audio_path: str) -> pretty_midi.PrettyMIDI:
    """
    Transcribe audio to MIDI using Spotify Basic Pitch.
    
    Args:
        audio_path: Path to audio file (MP3 or WAV)
    
    Returns:
        PrettyMIDI object with transcribed notes
    """
    model_output, midi_data, note_events = predict(audio_path)
    return midi_data

def load_midi_fallback(piece: str, recording: str) -> pretty_midi.PrettyMIDI:
    """
    Load pre-transcribed MIDI file as fallback.
    
    Use this when:
    - Basic Pitch fails
    - Demo needs to be fast and reliable
    - Testing without GPU
    """
    midi_path = MIDI_FALLBACKS[piece][recording]
    return pretty_midi.PrettyMIDI(midi_path)

async def get_midi_for_analysis(
    audio_path: str,
    piece: str,
    recording: str,
    use_fallback: bool = False
) -> pretty_midi.PrettyMIDI:
    """
    Get MIDI for analysis, with automatic fallback.
    
    In demo mode (use_fallback=True), skips transcription entirely
    for maximum reliability.
    """
    if use_fallback:
        return load_midi_fallback(piece, recording)
    
    try:
        return await transcribe_audio(audio_path)
    except Exception as e:
        print(f"Transcription failed: {e}, using fallback")
        return load_midi_fallback(piece, recording)
```

---

## Component 3: Score Alignment (DTW)

The key technical differentiator: **21.2% improvement** from score-aligned analysis.

```python
# analysis/alignment.py
import librosa
import numpy as np
import pretty_midi
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AlignmentResult:
    warping_path: np.ndarray
    alignment_cost: float
    time_mapping: List[Tuple[float, float]]  # (performance_time, reference_time)
    measure_boundaries: dict  # measure_num -> (start_time, end_time) in performance

def align_to_score(
    performance: pretty_midi.PrettyMIDI,
    reference: pretty_midi.PrettyMIDI,
    fs: int = 100  # Frames per second for piano roll
) -> AlignmentResult:
    """
    Align performance to reference score using Dynamic Time Warping.
    
    This enables measure-specific feedback by mapping performance time
    to score positions.
    """
    # Extract piano roll representations
    perf_roll = performance.get_piano_roll(fs=fs)
    ref_roll = reference.get_piano_roll(fs=fs)
    
    # Use chroma features for more robust alignment
    # (invariant to octave errors)
    perf_chroma = librosa.feature.chroma_cqt(
        y=np.sum(perf_roll, axis=0).astype(float),
        sr=fs
    )
    ref_chroma = librosa.feature.chroma_cqt(
        y=np.sum(ref_roll, axis=0).astype(float),
        sr=fs
    )
    
    # DTW alignment
    D, wp = librosa.sequence.dtw(perf_chroma, ref_chroma, subseq=True)
    
    # Create time mapping
    time_mapping = [(p[0] / fs, p[1] / fs) for p in wp]
    
    # Map measure boundaries from reference to performance
    measure_boundaries = map_measure_boundaries(wp, reference, fs)
    
    return AlignmentResult(
        warping_path=wp,
        alignment_cost=D[-1, -1],
        time_mapping=time_mapping,
        measure_boundaries=measure_boundaries
    )

def map_measure_boundaries(
    warping_path: np.ndarray,
    reference: pretty_midi.PrettyMIDI,
    fs: int
) -> dict:
    """
    Map measure numbers to time ranges in the performance.
    
    Uses the reference score's measure positions and warps them
    to performance time via the DTW alignment.
    """
    # Get measure boundaries from reference (requires MusicXML data)
    # For now, estimate from time signature and tempo
    # In production, load from MusicXML
    
    ref_duration = reference.get_end_time()
    # Assume 4/4 time, ~60 measures for Pathétique 3rd mvmt excerpt
    measures_estimate = 68
    measure_duration = ref_duration / measures_estimate
    
    boundaries = {}
    for m in range(1, measures_estimate + 1):
        ref_start = (m - 1) * measure_duration
        ref_end = m * measure_duration
        
        # Find corresponding performance times via warping path
        perf_start = warp_time(ref_start, warping_path, fs, direction="ref_to_perf")
        perf_end = warp_time(ref_end, warping_path, fs, direction="ref_to_perf")
        
        boundaries[m] = (perf_start, perf_end)
    
    return boundaries

def warp_time(
    time: float,
    warping_path: np.ndarray,
    fs: int,
    direction: str = "ref_to_perf"
) -> float:
    """Convert time from one alignment space to another."""
    frame = int(time * fs)
    
    if direction == "ref_to_perf":
        # Find warping path entry closest to reference frame
        ref_frames = warping_path[:, 1]
        idx = np.argmin(np.abs(ref_frames - frame))
        return warping_path[idx, 0] / fs
    else:
        perf_frames = warping_path[:, 0]
        idx = np.argmin(np.abs(perf_frames - frame))
        return warping_path[idx, 1] / fs
```

---

## Component 4: Performance Analysis

```python
# analysis/performance.py
import numpy as np
import pretty_midi
from dataclasses import dataclass
from typing import List, Optional
from .alignment import AlignmentResult

@dataclass
class MeasureAnalysis:
    measure_number: int
    timing_deviation_ms: float
    wrong_notes: List[int]
    velocity_variance: float
    issues: List[str]

@dataclass
class PerformanceAnalysis:
    overall_accuracy: float
    tempo_stability: float
    dynamic_range: float
    duration_seconds: float
    measures_analyzed: int
    measures_with_issues: List[MeasureAnalysis]
    strengths: List[str]

def analyze_performance_metrics(
    performance: pretty_midi.PrettyMIDI,
    reference: pretty_midi.PrettyMIDI,
    alignment: AlignmentResult
) -> PerformanceAnalysis:
    """
    Full performance analysis using score-aligned comparison.
    """
    perf_notes = performance.instruments[0].notes if performance.instruments else []
    ref_notes = reference.instruments[0].notes if reference.instruments else []
    
    measures = []
    timing_deviations = []
    wrong_notes_total = 0
    total_notes = 0
    
    for measure_num, (start, end) in alignment.measure_boundaries.items():
        # Get notes in this measure (performance)
        perf_measure_notes = [n for n in perf_notes if start <= n.start < end]
        
        # Get expected notes (reference) - use warped time
        ref_start, ref_end = get_reference_times(measure_num, alignment)
        ref_measure_notes = [n for n in ref_notes if ref_start <= n.start < ref_end]
        
        # Calculate metrics
        timing_dev = calculate_timing_deviation(perf_measure_notes, ref_measure_notes, alignment)
        wrong = find_wrong_notes(perf_measure_notes, ref_measure_notes)
        velocity_var = calculate_velocity_variance(perf_measure_notes)
        
        timing_deviations.append(timing_dev)
        wrong_notes_total += len(wrong)
        total_notes += len(ref_measure_notes)
        
        # Classify issues
        issues = []
        if abs(timing_dev) > 50:
            issues.append("rushing" if timing_dev < 0 else "dragging")
        if len(wrong) > 0:
            issues.append("wrong_notes")
        if velocity_var < 0.05:
            issues.append("flat_dynamics")
        if has_uneven_notes(perf_measure_notes):
            issues.append("uneven")
        
        if issues:  # Only include measures with issues
            measures.append(MeasureAnalysis(
                measure_number=measure_num,
                timing_deviation_ms=timing_dev,
                wrong_notes=wrong,
                velocity_variance=velocity_var,
                issues=issues
            ))
    
    # Aggregate metrics
    accuracy = 1 - (wrong_notes_total / total_notes) if total_notes > 0 else 1.0
    tempo_stability = 1 - min(1, np.std(timing_deviations) / 100)
    dynamic_range = calculate_dynamic_range(performance)
    
    # Identify strengths
    strengths = identify_strengths(
        accuracy, tempo_stability, dynamic_range,
        measures, performance
    )
    
    return PerformanceAnalysis(
        overall_accuracy=round(accuracy, 2),
        tempo_stability=round(max(0, tempo_stability), 2),
        dynamic_range=round(dynamic_range, 2),
        duration_seconds=round(performance.get_end_time(), 1),
        measures_analyzed=len(alignment.measure_boundaries),
        measures_with_issues=sorted(measures, key=lambda m: abs(m.timing_deviation_ms), reverse=True)[:10],
        strengths=strengths[:3]  # Top 3 strengths
    )

def identify_strengths(
    accuracy: float,
    tempo_stability: float,
    dynamic_range: float,
    measures: List[MeasureAnalysis],
    performance: pretty_midi.PrettyMIDI
) -> List[str]:
    """Identify genuine strengths to lead with in feedback."""
    strengths = []
    
    if accuracy > 0.95:
        strengths.append("Excellent note accuracy—nearly flawless")
    elif accuracy > 0.85:
        strengths.append("Strong note accuracy overall")
    
    if tempo_stability > 0.85:
        strengths.append("Steady, controlled tempo throughout")
    
    if dynamic_range > 0.7:
        strengths.append("Expressive dynamic range—you're making music, not just playing notes")
    
    # Check for good sections (measures without issues)
    issue_measures = {m.measure_number for m in measures}
    clean_measures = [m for m in range(1, 69) if m not in issue_measures]
    if len(clean_measures) > 50:
        strengths.append("Large sections are clean and controlled")
    
    if not strengths:
        strengths.append("Tackling a challenging piece with commitment")
    
    return strengths
```

---

## Component 5: Piece Data

```python
# data/pieces.py

PIECES = {
    "pathetique": {
        "id": "pathetique",
        "title": "Pathétique Sonata, 3rd Movement",
        "composer": "Ludwig van Beethoven",
        "opus": "Op. 13",
        "difficulty": "intermediate-advanced",
        "duration_typical": "4:30-5:30",
        "background": "Beethoven's most dramatic sonata, composed in 1798 during the onset of his hearing loss. The third movement is a rondo with relentless energy.",
        "listen_for": "Rhythmic precision in the rondo theme, dynamic contrasts, clean sixteenth-note passages",
        "common_issues": ["rushing sixteenth notes", "uneven left hand octaves", "losing pulse in transitions"],
        "reference_midi": "data/pathetique_reference.mid",
        "reference_xml": "data/pathetique_reference.xml"
    }
}

RECORDINGS = {
    "pathetique": {
        "amateur": {
            "label": "My Practice Recording",
            "description": "Jai's practice session with authentic mistakes",
            "audio_path": "data/pathetique_amateur.mp3",
            "midi_fallback_path": "data/pathetique_amateur.mid",
            "known_issues": ["rushing in development", "uneven runs", "dynamic inconsistencies"]
        },
        "professional": {
            "label": "Professional Reference",
            "description": "Clean performance for comparison",
            "audio_path": "data/pathetique_professional.mp3",
            "midi_fallback_path": "data/pathetique_professional.mid",
            "known_issues": []
        }
    }
}
```

---

## Deployment

### Modal Configuration

```python
# modal_app.py
import modal

app = modal.App("crescendai")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mcp",
        "fastapi",
        "uvicorn",
        "websockets",
        "basic-pitch",
        "pretty_midi",
        "music21",
        "librosa",
        "numpy",
        "pydantic",
    )
    .apt_install("ffmpeg")
)

@app.function(
    image=image,
    gpu="a10g",
    min_containers=2,
    enable_memory_snapshot=True,
    timeout=300,
    secrets=[modal.Secret.from_name("crescendai-secrets")],
)
@modal.asgi_app()
def serve():
    from mcp_server.server import app as fastapi_app
    return fastapi_app
```

### Pre-Warming Script

```python
# scripts/warm.py
"""Run every 5 minutes starting 30 min before demo."""
import asyncio
import aiohttp
import time

MCP_URL = "https://crescendai.modal.run"

async def warm():
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(f"{MCP_URL}/health") as resp:
            print(f"Health: {resp.status}")
        
        # Open WebSocket briefly
        async with session.ws_connect(f"{MCP_URL.replace('https', 'wss')}/ws/warmup") as ws:
            await ws.send_str('{"type": "ping"}')
            print("WebSocket warm")

if __name__ == "__main__":
    while True:
        asyncio.run(warm())
        time.sleep(300)
```

---

## File Structure

```
crescendai/
├── mcp_server/
│   ├── server.py            # FastAPI + MCP endpoints
│   ├── tools.py             # MCP tool definitions
│   ├── sessions.py          # Session management
│   ├── teaching.py          # Teaching context generator
│   ├── websocket.py         # WebSocket manager
│   └── __init__.py
├── analysis/
│   ├── transcription.py     # Audio → MIDI (with fallback)
│   ├── alignment.py         # DTW score alignment
│   ├── performance.py       # Performance metrics
│   └── __init__.py
├── companion/
│   ├── app/
│   │   ├── page.tsx         # Main companion page
│   │   └── layout.tsx
│   ├── components/
│   │   ├── SessionCode.tsx  # Session code display
│   │   ├── PianoRoll.tsx    # MIDI visualization
│   │   ├── RecordingSelect.tsx
│   │   └── StatusDisplay.tsx
│   ├── lib/
│   │   ├── websocket.ts     # WebSocket client
│   │   └── midi.ts          # MIDI playback
│   ├── package.json
│   └── next.config.js
├── data/
│   ├── pieces.py            # Piece metadata
│   ├── pathetique_reference.mid
│   ├── pathetique_reference.xml
│   ├── pathetique_amateur.mp3
│   ├── pathetique_amateur.mid
│   ├── pathetique_professional.mp3
│   └── pathetique_professional.mid
├── scripts/
│   ├── warm.py              # Pre-warming
│   └── test_flow.py         # Integration tests
├── modal_app.py
├── requirements.txt
└── README.md
```

---

## Latency Budget

| Step | Target | Notes |
|------|--------|-------|
| Session code spoken | ~1s | User reads 4 characters |
| Grok → MCP tool call | ~200ms | API overhead |
| MCP → Companion WebSocket | ~50ms | Persistent connection |
| Recording selection | User action | Click in companion |
| Analyze call → MCP | ~200ms | |
| Load audio + transcribe | 2-3s | Basic Pitch (or 0ms with fallback) |
| Alignment + Analysis | ~300ms | DTW + metrics |
| Generate teaching context | ~50ms | |
| Response → Grok | ~200ms | |
| Grok first token | ~350ms | Streaming begins |
| **Total (analyze → first feedback)** | **<4s** | With transcription |
| **Total (with MIDI fallback)** | **<1.5s** | Demo mode |

---

## Demo Checklist

### Day Before

- [ ] Record amateur audio (real practice with authentic mistakes)
- [ ] Obtain professional reference recording
- [ ] Pre-transcribe both to MIDI (fallback files)
- [ ] Verify Modal containers healthy (2 warm)
- [ ] Test full flow: Grok → MCP → Companion
- [ ] Pre-record backup video (720p)
- [ ] Practice pitch 10+ times
- [ ] Decide on demo session code (e.g., "PIANO")

### 30 Minutes Before

- [ ] Start warm.py script
- [ ] Open companion app, note session code
- [ ] Test Grok voice mode
- [ ] Verify network connectivity
- [ ] Phone hotspot ready as backup

### During Demo

1. Open companion app (shows session code)
2. "Hey Grok, start my piano lesson. Session code PIANO. I'm working on the Pathétique."
3. Wait for companion to show "✅ Connected"
4. Select "Amateur" recording in companion
5. "Analyze my performance"
6. Let Grok stream—don't interrupt
7. Watch companion react (highlights, playback)
8. Ask follow-up: "Why do I always rush there?"
9. Close with pitch line

### If Something Breaks

| Failure | Response |
|---------|----------|
| Session won't connect | "Let me refresh—" (reload companion) |
| Transcription slow/fails | MIDI fallback loads automatically |
| Grok doesn't call tools | Prompt explicitly: "Use your piano teacher tools to analyze this" |
| WebSocket drops | Companion auto-reconnects; regenerate code if needed |
| Total failure | "Let me show you a demo we recorded earlier" → backup video |

### Recovery Lines (Memorize These)

- "Give it a moment—it's analyzing 68 measures of Beethoven..."
- "Let me switch to the backup recording..."
- "The connection hiccuped—one sec while I reconnect..."
- "Here, let me show you what this looks like..." (→ video)
