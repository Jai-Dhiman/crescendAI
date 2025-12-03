"""
Modal deployment for CrescendAI MCP Server.
Deploy with: modal deploy modal_app.py
Test locally with: modal serve modal_app.py
"""

import modal

app = modal.App("crescendai-poc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mcp>=1.9.0",
        "uvicorn>=0.32.0",
        "starlette>=0.41.0",
    )
)


@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    import contextlib

    # Configure security to allow Modal hosts and xAI servers
    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "*.modal.run",
            "*.modal.run:*",
            "jai-dhiman--crescendai-poc-serve-dev.modal.run",
            "jai-dhiman--crescendai-poc-serve.modal.run",
        ],
        allowed_origins=["*"],
    )

    # Create the MCP server
    mcp = FastMCP("CrescendAI Piano Teacher", stateless_http=True, transport_security=security_settings)

    # Simulated session storage
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
        sessions[session_code] = {"piece": piece, "connected": True}

        return {
            "status": "connected",
            "piece_title": "Pathetique Sonata, 3rd Movement",
            "composer": "Ludwig van Beethoven",
            "message": f"Connected to session {session_code}. The Pathetique is loaded.",
            "teaching_context": {
                "piece_background": "Beethoven's most dramatic sonata, composed in 1798.",
                "what_to_listen_for": "Rhythmic precision, dynamic contrasts, clean sixteenth-note passages"
            }
        }

    @mcp.tool()
    def analyze_performance(session_code: str) -> dict:
        """
        Analyze the selected recording against the reference score.
        Call this when the user asks for feedback on their performance.
        """
        return {
            "analysis": {
                "overall_accuracy": 0.84,
                "tempo_stability": 0.71,
                "measures_with_issues": [
                    {"measure": 43, "issues": ["rushing"], "timing_deviation_ms": -89},
                    {"measure": 47, "issues": ["rushing", "uneven"], "timing_deviation_ms": -112},
                ],
                "strengths": ["Strong dynamic commitment in opening"]
            },
            "teaching_context": {
                "lead_with": "The dramatic opening chords show real commitment",
                "primary_focus": "Consistent rushing in measures 43-51",
                "analogy": "The left hand octaves should be like a heartbeat--steady and inevitable",
                "practice_strategy": "Isolate measures 43-51 at 60% tempo"
            }
        }

    @mcp.tool()
    def highlight_measures(session_code: str, measures: list[int], color: str = "red") -> dict:
        """Highlight specific measures on the companion app's piano roll."""
        return {
            "status": "highlighted",
            "measures": measures,
            "message": f"Measures {measures} highlighted in {color}."
        }

    @mcp.tool()
    def play_comparison(session_code: str, start_measure: int, end_measure: int) -> dict:
        """Play student's version and reference back-to-back for comparison."""
        return {
            "status": "playing_comparison",
            "message": f"Playing comparison for measures {start_measure}-{end_measure}."
        }

    @mcp.tool()
    def set_practice_tempo(session_code: str, tempo_percent: int) -> dict:
        """Set a practice tempo for the student."""
        return {
            "status": "tempo_set",
            "message": f"Practice tempo set to {tempo_percent}%."
        }

    @mcp.custom_route(path="/health", methods=["GET"])
    async def health_check(request: Request):
        return JSONResponse({"status": "healthy", "service": "crescendai-mcp"})

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[Mount("/", mcp.streamable_http_app())],
        lifespan=lifespan
    )

    return CORSMiddleware(
        starlette_app,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"]
    )
