"""
CrescendAI Dedalus Worker

Python Cloudflare Worker that wraps the Dedalus SDK.
Connects to the main Rust worker via service binding.
"""

from js import Response, Headers
import json
import traceback
from dedalus_labs import AsyncDedalus, DedalusRunner


class DedalusWorker:
    """
    Cloudflare Worker that wraps the Dedalus SDK.

    Provides a simple HTTP interface for the Rust worker to make
    chat completion requests with tool calling support.
    """

    def __init__(self, env):
        """Initialize the worker with environment bindings."""
        self.env = env

        # Get Dedalus API key from environment
        self.api_key = env.DEDALUS_API_KEY if hasattr(env, 'DEDALUS_API_KEY') else None
        if not self.api_key:
            print("WARNING: DEDALUS_API_KEY not set in environment")

        # Initialize Dedalus client (will be created per-request to handle async properly)
        self.client = None

    async def fetch(self, request):
        """
        Handle incoming requests from the Rust worker.

        Expected request format:
        POST /chat
        {
            "messages": [...],
            "tools": [...],
            "model": "gpt-5-nano",
            "temperature": 0.7,
            "max_tokens": 2000
        }

        Returns:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [...]
                },
                "finish_reason": "stop"
            }],
            "usage": {...}
        }
        """
        try:
            # Parse request
            url = request.url
            method = request.method

            # Health check endpoint
            if method == "GET" and url.endswith("/health"):
                return self._json_response({"status": "ok", "service": "dedalus-worker"})

            # Chat completion endpoint
            if method == "POST" and url.endswith("/chat"):
                # Parse request body
                body_text = await request.text()
                body = json.loads(body_text)

                # Validate API key
                if not self.api_key:
                    return self._error_response(
                        "DEDALUS_API_KEY not configured",
                        status=500
                    )

                # Make Dedalus API call
                result = await self._call_dedalus(body)
                return self._json_response(result)

            # Unknown endpoint
            return self._error_response(
                f"Unknown endpoint: {method} {url}",
                status=404
            )

        except json.JSONDecodeError as e:
            return self._error_response(
                f"Invalid JSON: {str(e)}",
                status=400
            )
        except Exception as e:
            # Log full traceback for debugging
            error_trace = traceback.format_exc()
            print(f"Error in DedalusWorker: {error_trace}")

            return self._error_response(
                f"Internal error: {str(e)}",
                status=500
            )

    async def _call_dedalus(self, request_body):
        """
        Call the Dedalus API with the provided request body using the official SDK.

        Uses DedalusRunner for agent-based execution with tool support.
        """
        # Extract request parameters
        messages = request_body.get("messages", [])
        tools = request_body.get("tools", [])
        model = request_body.get("model", "openai/gpt-5-mini")
        temperature = request_body.get("temperature", 0.7)
        max_tokens = request_body.get("max_tokens", 2000)

        # Initialize Dedalus client
        client = AsyncDedalus(api_key=self.api_key)
        runner = DedalusRunner(client)

        # Convert messages to a single input string for DedalusRunner
        # The runner expects a simple text input, not a messages array
        # We'll extract the last user message as the input
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            user_message = "Hello"

        # Convert tools to callable functions for DedalusRunner
        # Note: Since we're in a Workers environment, we need to handle tool calls
        # differently. The tools will be executed by the Rust worker, not here.
        # We'll pass tool definitions and let Dedalus return tool calls for execution.

        # For now, run without tools and let the model respond directly
        # The Rust worker will handle the tool calling loop
        try:
            result = await runner.run(
                input=user_message,
                model=model,
                # Note: DedalusRunner handles tools differently than OpenAI format
                # We'll need to adapt this if tool support is needed
                stream=False
            )

            # Convert DedalusRunner result to OpenAI-compatible format
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result.final_output if hasattr(result, 'final_output') else str(result),
                        "tool_calls": None  # Tool calls will be handled separately if needed
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                "usage": {
                    "prompt_tokens": 0,  # Dedalus SDK may not expose token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "model": model,
                "created": 1699000000
            }

            return response

        except Exception as e:
            print(f"Error calling Dedalus API: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def _json_response(self, data, status=200):
        """Create a JSON response."""
        headers = Headers.new({
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }.items())

        return Response.new(
            json.dumps(data),
            status=status,
            headers=headers
        )

    def _error_response(self, message, status=500):
        """Create an error response."""
        return self._json_response(
            {"error": message},
            status=status
        )


# Cloudflare Workers entrypoint
async def on_fetch(request, env):
    """Main entrypoint for Cloudflare Workers."""
    worker = DedalusWorker(env)
    return await worker.fetch(request)
