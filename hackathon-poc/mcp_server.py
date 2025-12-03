"""
Minimal MCP Server for CrescendAI POC.
Tests if we can create an MCP server that Grok can call.
"""

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import contextlib

# Create the MCP server
mcp = FastMCP("CrescendAI Piano Teacher", stateless_http=True)

# Simulated session storage (in production, use Redis or similar)
sessions = {}


@mcp.tool()
def start_lesson(session_code: str, piece: str = "pathetique") -> dict:
    """
    Connect to a companion app session and load a piece for the lesson.
    Call this when the user provides their session code and piece name.

    Args:
        session_code: The 4-character code displayed on the companion app
        piece: The piece to practice (e.g., "pathetique")

    Returns:
        Connection status and piece information with teaching context
    """
    session_code = session_code.upper()

    # Store session
    sessions[session_code] = {
        "piece": piece,
        "connected": True,
        "recording": None
    }

    return {
        "status": "connected",
        "piece_title": "Pathetique Sonata, 3rd Movement",
        "composer": "Ludwig van Beethoven",
        "message": f"Connected to session {session_code}. The Pathetique is loaded. Select your recording in the companion app.",
        "teaching_context": {
            "piece_background": "Beethoven's most dramatic sonata, composed in 1798 during the onset of his hearing loss.",
            "what_to_listen_for": "Rhythmic precision in the rondo theme, dynamic contrasts, clean sixteenth-note passages"
        }
    }


@mcp.tool()
def analyze_performance(session_code: str) -> dict:
    """
    Analyze the selected recording against the reference score.
    Call this when the user asks for feedback on their performance.

    Returns detailed analysis plus teaching hints to guide your response.
    Lead with encouragement, then address the primary focus area.

    Args:
        session_code: The session code for the active lesson
    """
    session_code = session_code.upper()

    if session_code not in sessions:
        return {
            "status": "error",
            "message": f"Session {session_code} not found. Ask the user to check their session code."
        }

    # Simulated analysis results (in production, this would be real audio analysis)
    return {
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
            "lead_with": "The dramatic opening chords show real commitment--this student understands the character",
            "primary_focus": "Consistent rushing in measures 43-51 (sixteenth-note passages)",
            "root_cause_hint": "Likely anxiety about the technical difficulty causing 'survival mode'",
            "analogy": "The left hand octaves should be like a heartbeat--steady and inevitable, grounding the drama",
            "practice_strategy": "Isolate measures 43-51 at 60% tempo, left hand alone first",
            "personality_note": "Be encouraging but specific. Use the heartbeat analogy."
        }
    }


@mcp.tool()
def highlight_measures(session_code: str, measures: list[int], color: str = "red") -> dict:
    """
    Highlight specific measures on the companion app's piano roll.
    Use this to draw attention to problem areas while explaining them.

    Args:
        session_code: The session code
        measures: List of measure numbers to highlight
        color: Color for highlighting - "red" (problems), "yellow" (attention), "green" (good)
    """
    return {
        "status": "highlighted",
        "measures": measures,
        "color": color,
        "message": f"Measures {measures} are now highlighted in {color} on the student's screen."
    }


@mcp.tool()
def play_comparison(session_code: str, start_measure: int, end_measure: int) -> dict:
    """
    Play the student's version and reference back-to-back for comparison.
    Highly effective for demonstrating timing and accuracy differences.

    Args:
        session_code: The session code
        start_measure: First measure to play
        end_measure: Last measure to play
    """
    return {
        "status": "playing_comparison",
        "message": f"Playing comparison (student then reference) for measures {start_measure}-{end_measure}. The difference should be audible."
    }


@mcp.tool()
def set_practice_tempo(session_code: str, tempo_percent: int) -> dict:
    """
    Set a practice tempo for the student to try.
    Use this when suggesting they slow down a difficult passage.

    Args:
        session_code: The session code
        tempo_percent: Percentage of original tempo (e.g., 60 for 60%)
    """
    return {
        "status": "tempo_set",
        "tempo_percent": tempo_percent,
        "message": f"Practice tempo set to {tempo_percent}%. The student can now play along with the slowed reference."
    }


# Custom health check endpoint
@mcp.custom_route(path="/health", methods=["GET"])
async def health_check(request: Request):
    return JSONResponse({"status": "healthy", "service": "crescendai-mcp"})


# Build the app with CORS for browser clients
@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Mount("/", mcp.streamable_http_app())
    ],
    lifespan=lifespan
)

app = CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"]
)


if __name__ == "__main__":
    import uvicorn
    print("Starting CrescendAI MCP Server on http://localhost:8000")
    print("MCP endpoint: http://localhost:8000/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8000)
