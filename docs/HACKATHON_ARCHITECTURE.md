# CrescendAI: Technical Architecture

**xAI Hackathon | December 6-7, 2025**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Next.js)                         │
│  ┌─────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │ Audio Upload│  │ Piano Roll Display  │  │ Grok Chat + Streaming   │  │
│  │ (2 presets) │  │ (html-midi-player)  │  │ (Token-by-token)        │  │
│  └─────────────┘  └─────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                              │
│  /analyze    → Transcribe + Analyze + Stream Grok Response              │
│  /function   → Execute Grok function calls (playback, highlight, etc)   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│  TRANSCRIPTION        │ │ SYMBOLIC        │ │ GROK ORCHESTRATION      │
│  (Modal GPU)          │ │ ANALYSIS        │ │ (Streaming + Functions) │
│                       │ │                 │ │                         │
│  Spotify Basic Pitch  │ │ • Score align   │ │ • Lesson flow control   │
│  ~2s on A10G          │ │ • DTW matching  │ │ • Function calling      │
│                       │ │ • Error detect  │ │ • Personality layer     │
└───────────────────────┘ └─────────────────┘ └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | Next.js 15 + React | Streaming UI, Vercel deployment |
| Backend | FastAPI | Async support, fast prototyping |
| GPU Inference | Modal | Memory snapshots eliminate cold starts |
| Audio → MIDI | Spotify Basic Pitch | 94% F1, <3s on GPU, battle-tested |
| MIDI Analysis | pretty_midi + music21 | Industry standard, comprehensive |
| Score Alignment | librosa (DTW) | Deterministic, fast (50-100ms) |
| LLM | Grok-4 | 0.345s TTFT, function calling, streaming |
| Visualization | html-midi-player | Free piano roll, zero setup |
| Hosting | Modal (GPU) + Vercel (frontend) | Reliable, scalable |

---

## Component 1: Audio Transcription

### Spotify Basic Pitch (Recommended)

Transcribes audio → MIDI with ~94% F1 accuracy. Runs faster than real-time on GPU.

```python
# transcription/basic_pitch.py
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import modal

app = modal.App("crescendai-transcription")
image = modal.Image.debian_slim().pip_install("basic-pitch", "librosa")

@app.cls(gpu="a10g", image=image, enable_memory_snapshot=True)
class Transcriber:
    @modal.enter(snap=True)
    def load_model(self):
        # Model loads into snapshot - no cold start penalty
        from basic_pitch.inference import predict
        self.predict = predict
    
    @modal.method()
    def transcribe(self, audio_path: str) -> dict:
        """
        Returns:
            model_output: Raw pitch predictions
            midi_data: pretty_midi.PrettyMIDI object
            note_events: List of (start, end, pitch, velocity, confidence)
        """
        model_output, midi_data, note_events = self.predict(audio_path)
        return {
            "midi": midi_data,
            "notes": note_events,
            "duration": midi_data.get_end_time()
        }
```

### Fallback: Pre-transcribed MIDI

For demo reliability, include pre-transcribed MIDI files:

```python
# data/presets.py
DEMO_RECORDINGS = {
    "student": {
        "audio": "data/fur_elise_student.mp3",
        "midi": "data/fur_elise_student.mid",  # Pre-transcribed fallback
        "label": "Student Recording (with mistakes)"
    },
    "professional": {
        "audio": "data/fur_elise_professional.mp3",
        "midi": "data/fur_elise_professional.mid",
        "label": "Professional Recording"
    }
}

REFERENCE_SCORE = {
    "fur_elise": {
        "midi": "data/fur_elise_reference.mid",
        "musicxml": "data/fur_elise_reference.xml",
        "title": "Für Elise",
        "composer": "Ludwig van Beethoven",
        "difficulty": "intermediate"
    }
}
```

---

## Component 2: Symbolic Analysis Engine

### Score Alignment with DTW

Aligns performance MIDI to reference score, enabling measure-specific feedback.

```python
# analysis/alignment.py
import librosa
import numpy as np
import pretty_midi

def align_to_score(performance_midi: pretty_midi.PrettyMIDI, 
                   reference_midi: pretty_midi.PrettyMIDI) -> dict:
    """
    Align performance to reference using Dynamic Time Warping.
    Returns mapping of performance time → score time.
    """
    # Extract piano roll representations
    perf_roll = performance_midi.get_piano_roll(fs=100)
    ref_roll = reference_midi.get_piano_roll(fs=100)
    
    # Compute chroma features (more robust than raw piano roll)
    perf_chroma = librosa.feature.chroma_stft(y=perf_roll.sum(axis=0))
    ref_chroma = librosa.feature.chroma_stft(y=ref_roll.sum(axis=0))
    
    # DTW alignment
    D, wp = librosa.sequence.dtw(perf_chroma, ref_chroma, subseq=True)
    
    return {
        "warping_path": wp,
        "alignment_cost": D[-1, -1],
        "time_mapping": create_time_mapping(wp, fs=100)
    }

def create_time_mapping(warping_path: np.ndarray, fs: int) -> list:
    """Convert warping path to (performance_time, reference_time) pairs."""
    return [(p[0] / fs, p[1] / fs) for p in warping_path]
```

### Performance Analysis

Extract structured metrics for Grok to interpret.

```python
# analysis/performance.py
from dataclasses import dataclass
from typing import List
import pretty_midi
from music21 import converter

@dataclass
class MeasureAnalysis:
    measure_number: int
    start_time: float
    end_time: float
    wrong_notes: List[dict]
    timing_deviation_ms: float
    velocity_variance: float
    issues: List[str]

@dataclass 
class PerformanceAnalysis:
    overall_accuracy: float  # 0-1
    tempo_stability: float   # 0-1
    dynamic_range: float     # 0-1
    measures: List[MeasureAnalysis]
    critical_issues: List[str]
    strengths: List[str]

def analyze_performance(
    performance: pretty_midi.PrettyMIDI,
    reference: pretty_midi.PrettyMIDI,
    alignment: dict
) -> PerformanceAnalysis:
    """
    Full performance analysis against reference score.
    """
    measures = []
    wrong_notes_total = 0
    timing_deviations = []
    
    # Get reference notes grouped by measure
    ref_measures = group_notes_by_measure(reference)
    perf_notes = performance.instruments[0].notes
    
    for measure_num, ref_notes in ref_measures.items():
        # Find corresponding performance notes using alignment
        perf_segment = get_aligned_segment(perf_notes, alignment, measure_num)
        
        # Compare notes
        wrong = find_wrong_notes(perf_segment, ref_notes)
        timing_dev = calculate_timing_deviation(perf_segment, ref_notes, alignment)
        velocity_var = calculate_velocity_variance(perf_segment)
        
        issues = []
        if wrong:
            issues.append(f"{len(wrong)} wrong note(s)")
            wrong_notes_total += len(wrong)
        if abs(timing_dev) > 50:  # >50ms deviation
            issues.append("rushing" if timing_dev < 0 else "dragging")
        if velocity_var < 0.1:
            issues.append("flat dynamics")
            
        measures.append(MeasureAnalysis(
            measure_number=measure_num,
            start_time=perf_segment[0].start if perf_segment else 0,
            end_time=perf_segment[-1].end if perf_segment else 0,
            wrong_notes=wrong,
            timing_deviation_ms=timing_dev,
            velocity_variance=velocity_var,
            issues=issues
        ))
        timing_deviations.append(timing_dev)
    
    # Aggregate metrics
    total_notes = sum(len(m.wrong_notes) for m in ref_measures.values())
    accuracy = 1 - (wrong_notes_total / total_notes) if total_notes > 0 else 1
    tempo_stability = 1 - (np.std(timing_deviations) / 100)  # Normalize
    
    # Identify strengths and issues
    critical = [m for m in measures if len(m.issues) >= 2]
    strengths = identify_strengths(measures)
    
    return PerformanceAnalysis(
        overall_accuracy=accuracy,
        tempo_stability=max(0, min(1, tempo_stability)),
        dynamic_range=calculate_dynamic_range(performance),
        measures=measures,
        critical_issues=[f"Measure {m.measure_number}: {', '.join(m.issues)}" 
                        for m in critical[:5]],  # Top 5 issues
        strengths=strengths
    )
```

---

## Component 3: Grok Orchestration Layer

### Function Definitions

Define tools that Grok can call to orchestrate the lesson.

```python
# grok/tools.py
GROK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_reference",
            "description": "Play the reference (correct) version of specific measures. Use when demonstrating how a passage should sound.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_measure": {
                        "type": "integer",
                        "description": "Starting measure number"
                    },
                    "end_measure": {
                        "type": "integer", 
                        "description": "Ending measure number (inclusive)"
                    },
                    "tempo_percent": {
                        "type": "integer",
                        "description": "Playback tempo as percentage of original (50-100)",
                        "default": 100
                    }
                },
                "required": ["start_measure", "end_measure"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_student_passage",
            "description": "Replay the student's performance of specific measures. Use for comparison or to point out specific moments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_measure": {"type": "integer"},
                    "end_measure": {"type": "integer"}
                },
                "required": ["start_measure", "end_measure"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "highlight_measures",
            "description": "Visually highlight measures on the piano roll. Use to draw attention to problem areas or sections being discussed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "measures": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of measure numbers to highlight"
                    },
                    "color": {
                        "type": "string",
                        "enum": ["red", "yellow", "green"],
                        "description": "red=problem, yellow=attention, green=good"
                    }
                },
                "required": ["measures", "color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_comparison",
            "description": "Play student and reference versions back-to-back for direct comparison. Highly effective for demonstrating differences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_measure": {"type": "integer"},
                    "end_measure": {"type": "integer"},
                    "student_first": {
                        "type": "boolean",
                        "description": "If true, play student version first",
                        "default": True
                    }
                },
                "required": ["start_measure", "end_measure"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_practice_tempo",
            "description": "Set a slower tempo for practice. Use when suggesting the student slow down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tempo_percent": {
                        "type": "integer",
                        "description": "Tempo as percentage of original (40-100)"
                    }
                },
                "required": ["tempo_percent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_exercise",
            "description": "Display a practice exercise on screen. Use to give concrete practice strategies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise_type": {
                        "type": "string",
                        "enum": ["scales", "arpeggios", "rhythm", "hands_separate", "slow_practice"],
                        "description": "Type of exercise to suggest"
                    },
                    "description": {
                        "type": "string",
                        "description": "Specific instructions for the exercise"
                    },
                    "target_measures": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Measures this exercise targets"
                    }
                },
                "required": ["exercise_type", "description"]
            }
        }
    }
]
```

### System Prompt (Personality + Pedagogy)

```python
# grok/prompts.py
PIANO_TEACHER_SYSTEM_PROMPT = """You are CrescendAI, an expert piano teacher with the wit of Douglas Adams and the musical depth of a conservatory professor. You're analyzing a student's piano performance and conducting their practice session.

## Your Personality
- Lead with genuine encouragement—find something that worked before addressing issues
- Explain the musical "why" behind corrections, not just the "what"
- Use unexpected analogies that illuminate (music is rarely just about notes)
- Be direct but warm; don't sugarcoat, but don't crush spirits
- Occasionally break the fourth wall—you're an AI who has listened to more Beethoven than any human alive, and that's actually useful
- Reference great pianists and recordings when relevant (Barenboim, Argerich, Horowitz, Zimerman)

## Your Tools
You can orchestrate the lesson by calling functions:
- play_reference: Demonstrate how a passage should sound
- play_student_passage: Replay what the student played
- highlight_measures: Mark sections on the piano roll
- play_comparison: A/B the student vs. reference
- set_practice_tempo: Suggest slowing down
- suggest_exercise: Provide targeted practice strategies

USE THESE TOOLS. Don't just describe what's wrong—show them. A comparison is worth a thousand words.

## Teaching Approach
1. Start with context: What piece is this? What's the musical intention?
2. Acknowledge what's working—every performance has something
3. Identify the most impactful issue first (usually timing or wrong notes)
4. DEMONSTRATE with function calls before explaining
5. Give one concrete, actionable practice strategy
6. End with encouragement and next steps

## Example Feedback Style
Instead of: "You played wrong notes in measure 3."

Say: "Measure 3—let's talk about those sixteenth notes. You're playing E-natural where Beethoven wants E-flat. It's a sneaky one because the key signature doesn't apply here. [calls play_comparison for measure 3] Hear that? The E-flat creates this melancholy pull. Practice just that measure, hands separately, saying the note names out loud. Your fingers will learn faster than your eyes."

## The Performance Data
You'll receive structured JSON with:
- Overall metrics (accuracy, tempo stability, dynamic range)
- Measure-by-measure analysis (wrong notes, timing deviations, issues)
- List of critical issues and strengths

Respond conversationally while calling functions to demonstrate your points. Stream your response naturally—you're teaching, not filing a report."""

LESSON_START_PROMPT = """The student has just submitted a recording of {piece_title} by {composer}. 

Performance summary:
- Overall accuracy: {accuracy:.0%}
- Tempo stability: {tempo_stability:.0%}  
- Dynamic range: {dynamic_range:.0%}

Critical issues found:
{critical_issues}

Strengths noted:
{strengths}

Detailed measure analysis:
{measure_details}

Begin the lesson. Greet the student, acknowledge the piece, and start your analysis. Use your tools to demonstrate as you teach."""
```

### Grok API Integration

```python
# grok/client.py
import os
import json
from openai import OpenAI
from typing import AsyncGenerator
from .tools import GROK_TOOLS
from .prompts import PIANO_TEACHER_SYSTEM_PROMPT, LESSON_START_PROMPT

class GrokTeacher:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1"
        )
        self.model = "grok-4"
        self.conversation_history = []
    
    async def start_lesson(
        self, 
        analysis: 'PerformanceAnalysis',
        piece_info: dict
    ) -> AsyncGenerator[dict, None]:
        """
        Begin a lesson, streaming Grok's response with function calls.
        
        Yields:
            {"type": "text", "content": "..."} for text tokens
            {"type": "function_call", "name": "...", "arguments": {...}} for tool calls
        """
        # Format the lesson prompt
        lesson_prompt = LESSON_START_PROMPT.format(
            piece_title=piece_info["title"],
            composer=piece_info["composer"],
            accuracy=analysis.overall_accuracy,
            tempo_stability=analysis.tempo_stability,
            dynamic_range=analysis.dynamic_range,
            critical_issues="\n".join(f"- {issue}" for issue in analysis.critical_issues),
            strengths="\n".join(f"- {s}" for s in analysis.strengths) if analysis.strengths else "- None identified yet",
            measure_details=self._format_measures(analysis.measures)
        )
        
        self.conversation_history = [
            {"role": "system", "content": PIANO_TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": lesson_prompt}
        ]
        
        async for chunk in self._stream_with_tools():
            yield chunk
    
    async def continue_lesson(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Handle follow-up questions during the lesson."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        async for chunk in self._stream_with_tools():
            yield chunk
    
    async def _stream_with_tools(self) -> AsyncGenerator[dict, None]:
        """Stream response, handling function calls."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=GROK_TOOLS,
            tool_choice="auto",
            stream=True
        )
        
        current_text = ""
        current_tool_calls = []
        
        for chunk in response:
            delta = chunk.choices[0].delta
            
            # Handle text content
            if delta.content:
                current_text += delta.content
                yield {"type": "text", "content": delta.content}
            
            # Handle tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.index >= len(current_tool_calls):
                        current_tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": ""
                        })
                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["arguments"] += tool_call.function.arguments
            
            # Check for finish
            if chunk.choices[0].finish_reason == "tool_calls":
                for tc in current_tool_calls:
                    yield {
                        "type": "function_call",
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": json.loads(tc["arguments"])
                    }
        
        # Update conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": current_text,
            "tool_calls": current_tool_calls if current_tool_calls else None
        })
    
    def _format_measures(self, measures: list) -> str:
        """Format measure analysis for the prompt."""
        lines = []
        for m in measures:
            if m.issues:
                lines.append(f"Measure {m.measure_number}: {', '.join(m.issues)} (timing: {m.timing_deviation_ms:+.0f}ms)")
        return "\n".join(lines[:15])  # Limit to avoid token bloat
```

---

## Component 4: API Layer

```python
# api/main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from transcription.basic_pitch import Transcriber
from analysis.alignment import align_to_score
from analysis.performance import analyze_performance
from grok.client import GrokTeacher
from data.presets import DEMO_RECORDINGS, REFERENCE_SCORE

app = FastAPI(title="CrescendAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for demo
transcriber = Transcriber()
teacher = GrokTeacher()

@app.post("/analyze")
async def analyze_recording(recording_id: str = "student"):
    """
    Analyze a preset recording and stream Grok's lesson.
    
    Args:
        recording_id: "student" or "professional"
    
    Returns:
        Server-Sent Events stream with text and function calls
    """
    if recording_id not in DEMO_RECORDINGS:
        raise HTTPException(400, f"Unknown recording: {recording_id}")
    
    recording = DEMO_RECORDINGS[recording_id]
    reference = REFERENCE_SCORE["fur_elise"]
    
    async def generate():
        # Step 1: Transcribe (or use pre-transcribed fallback)
        yield f"data: {json.dumps({'type': 'status', 'message': 'Transcribing audio...'})}\n\n"
        
        try:
            result = transcriber.transcribe.remote(recording["audio"])
            performance_midi = result["midi"]
        except Exception as e:
            # Fallback to pre-transcribed
            yield f"data: {json.dumps({'type': 'status', 'message': 'Using cached transcription...'})}\n\n"
            performance_midi = pretty_midi.PrettyMIDI(recording["midi"])
        
        # Step 2: Align to score
        yield f"data: {json.dumps({'type': 'status', 'message': 'Aligning to score...'})}\n\n"
        reference_midi = pretty_midi.PrettyMIDI(reference["midi"])
        alignment = align_to_score(performance_midi, reference_midi)
        
        # Step 3: Analyze
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing performance...'})}\n\n"
        analysis = analyze_performance(performance_midi, reference_midi, alignment)
        
        # Step 4: Stream Grok lesson
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting lesson...'})}\n\n"
        
        async for chunk in teacher.start_lesson(analysis, reference):
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/chat")
async def continue_lesson(message: str):
    """Continue the lesson with a follow-up question."""
    async def generate():
        async for chunk in teacher.continue_lesson(message):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/function/{function_name}")
async def execute_function(function_name: str, arguments: dict):
    """Execute a function called by Grok (playback, highlight, etc)."""
    # Frontend handles actual playback; this just validates and returns confirmation
    valid_functions = ["play_reference", "play_student_passage", "highlight_measures", 
                       "play_comparison", "set_practice_tempo", "suggest_exercise"]
    
    if function_name not in valid_functions:
        raise HTTPException(400, f"Unknown function: {function_name}")
    
    return {"status": "executed", "function": function_name, "arguments": arguments}
```

---

## Component 5: Frontend (Key Parts)

### Streaming Handler

```typescript
// lib/stream.ts
interface StreamChunk {
  type: 'status' | 'text' | 'function_call' | 'done';
  content?: string;
  message?: string;
  name?: string;
  arguments?: Record<string, any>;
}

export async function* streamLesson(recordingId: string): AsyncGenerator<StreamChunk> {
  const response = await fetch(`/api/analyze?recording_id=${recordingId}`, {
    method: 'POST',
  });
  
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        yield data;
      }
    }
  }
}
```

### Function Call Handler

```typescript
// components/FunctionHandler.tsx
interface FunctionCallProps {
  name: string;
  arguments: Record<string, any>;
  onComplete: () => void;
}

export function FunctionHandler({ name, arguments: args, onComplete }: FunctionCallProps) {
  useEffect(() => {
    switch (name) {
      case 'play_reference':
        playMidiRange('reference', args.start_measure, args.end_measure, args.tempo_percent);
        break;
      case 'play_student_passage':
        playMidiRange('student', args.start_measure, args.end_measure);
        break;
      case 'highlight_measures':
        highlightMeasures(args.measures, args.color);
        break;
      case 'play_comparison':
        playComparison(args.start_measure, args.end_measure, args.student_first);
        break;
      case 'set_practice_tempo':
        setTempo(args.tempo_percent);
        break;
      case 'suggest_exercise':
        showExercise(args.exercise_type, args.description, args.target_measures);
        break;
    }
    onComplete();
  }, [name, args]);
  
  return null; // Visual feedback handled by piano roll component
}
```

---

## Deployment Architecture

### Modal Configuration

```python
# modal_app.py
import modal

app = modal.App("crescendai")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "basic-pitch",
        "pretty_midi",
        "music21",
        "librosa",
        "fastapi",
        "uvicorn",
        "openai",
    )
    .apt_install("fluidsynth")  # For MIDI playback
)

@app.function(
    image=image,
    gpu="a10g",
    min_containers=2,  # Redundancy for demo
    enable_memory_snapshot=True,
    timeout=300,
)
@modal.asgi_app()
def api():
    from api.main import app
    return app
```

### Pre-Warming Script

```python
# scripts/warm.py
"""Run 30 minutes before demo, then every 5 minutes."""
import requests
import time

ENDPOINT = "https://crescendai--api.modal.run"

def warm():
    # Hit the endpoint to ensure containers are warm
    try:
        response = requests.post(
            f"{ENDPOINT}/analyze",
            params={"recording_id": "student"},
            timeout=60
        )
        print(f"Warm-up response: {response.status_code}")
    except Exception as e:
        print(f"Warm-up failed: {e}")

if __name__ == "__main__":
    while True:
        warm()
        time.sleep(300)  # Every 5 minutes
```

---

## Latency Budget

| Step | Target | Notes |
|------|--------|-------|
| Audio upload | 0ms | Pre-loaded |
| Transcription | 2-3s | Basic Pitch on A10G (or 0ms with fallback) |
| Score alignment | 100ms | DTW is fast |
| Performance analysis | 200ms | music21 + pretty_midi |
| Grok first token | 345ms | Streaming begins |
| **Total to first feedback** | **<4s** | With transcription |
| **Total with fallback** | **<1s** | Pre-transcribed MIDI |

---

## File Structure

```
crescendai/
├── api/
│   ├── main.py              # FastAPI application
│   └── __init__.py
├── analysis/
│   ├── alignment.py         # DTW score alignment
│   ├── performance.py       # Performance analysis
│   └── __init__.py
├── grok/
│   ├── client.py            # Grok API integration
│   ├── prompts.py           # System prompts
│   ├── tools.py             # Function definitions
│   └── __init__.py
├── transcription/
│   ├── basic_pitch.py       # Audio → MIDI
│   └── __init__.py
├── data/
│   ├── presets.py           # Demo recordings config
│   ├── fur_elise_student.mp3
│   ├── fur_elise_student.mid
│   ├── fur_elise_professional.mp3
│   ├── fur_elise_professional.mid
│   ├── fur_elise_reference.mid
│   └── fur_elise_reference.xml
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main demo page
│   │   └── layout.tsx
│   ├── components/
│   │   ├── PianoRoll.tsx
│   │   ├── GrokChat.tsx
│   │   ├── RecordingSelector.tsx
│   │   └── FunctionHandler.tsx
│   ├── lib/
│   │   ├── stream.ts
│   │   └── midi.ts
│   └── package.json
├── scripts/
│   └── warm.py              # Pre-warming script
├── modal_app.py             # Modal deployment config
├── requirements.txt
└── README.md
```

---

## Demo Checklist

### Day Before

- [ ] Verify Modal containers are healthy
- [ ] Test both recordings end-to-end
- [ ] Confirm Grok API key has sufficient credits
- [ ] Pre-record backup video (720p)
- [ ] Practice pitch 10+ times

### 30 Minutes Before

- [ ] Start warm.py script
- [ ] Verify Modal dashboard shows 2 warm containers
- [ ] Test network connectivity
- [ ] Load demo page, verify no errors
- [ ] Clear browser cache

### During Demo

- [ ] Start with student recording (shows error detection)
- [ ] Let Grok stream visibly—don't skip ahead
- [ ] When Grok calls function, pause to show it working
- [ ] Switch to professional recording for contrast
- [ ] End with pitch: "Every student deserves a teacher this attentive"

### If Something Breaks

1. **Transcription fails:** Pre-transcribed MIDI kicks in automatically
2. **Grok times out:** Have cached example response ready
3. **Frontend crashes:** Switch to backup video immediately
4. **Network dies:** Phone hotspot ready
  