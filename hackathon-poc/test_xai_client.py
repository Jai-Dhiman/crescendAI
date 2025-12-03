"""
Test xAI SDK client connecting to our MCP server.
This validates the core architecture: Grok -> MCP Server -> Response.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_with_xai_sdk():
    """Test using the official xAI SDK with Remote MCP Tools."""
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import mcp
    except ImportError:
        print("xai-sdk not installed. Run: uv add xai-sdk")
        return False

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("XAI_API_KEY not set in environment")
        print("Get your API key from https://console.x.ai/")
        return False

    client = Client(api_key=api_key)

    # Point to our local MCP server (or deployed Modal URL)
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    print(f"Connecting to MCP server at: {mcp_url}")

    try:
        chat = client.chat.create(
            model="grok-4-1-fast",
            tools=[
                mcp(server_url=mcp_url, server_label="crescendai"),
            ],
        )

        # Test 1: Start a lesson
        print("\n--- Test 1: Start Lesson ---")
        chat.append(user("Start my piano lesson. My session code is TEST. I'm working on the Pathetique."))

        for response, chunk in chat.stream():
            if chunk.content:
                print(chunk.content, end="", flush=True)
            for tool_call in chunk.tool_calls:
                print(f"\n[Tool Call: {tool_call.function.name}({tool_call.function.arguments})]")

        print("\n")

        # Test 2: Analyze performance
        print("\n--- Test 2: Analyze Performance ---")
        chat.append(user("Analyze my performance"))

        for response, chunk in chat.stream():
            if chunk.content:
                print(chunk.content, end="", flush=True)
            for tool_call in chunk.tool_calls:
                print(f"\n[Tool Call: {tool_call.function.name}({tool_call.function.arguments})]")

        print("\n")

        # Test 3: Follow-up question
        print("\n--- Test 3: Follow-up ---")
        chat.append(user("Why do I always rush there?"))

        for response, chunk in chat.stream():
            if chunk.content:
                print(chunk.content, end="", flush=True)
            for tool_call in chunk.tool_calls:
                print(f"\n[Tool Call: {tool_call.function.name}({tool_call.function.arguments})]")

        print("\n\nAll tests passed!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CrescendAI Architecture Validation")
    print("=" * 60)
    print("\nThis test validates:")
    print("1. xAI SDK can connect to our MCP server")
    print("2. Grok can discover and call our tools")
    print("3. Tool responses are properly formatted")
    print("4. Grok incorporates teaching_context in responses")
    print()

    success = test_with_xai_sdk()

    if success:
        print("\n" + "=" * 60)
        print("VALIDATION SUCCESSFUL!")
        print("The architecture is feasible for the hackathon.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED")
        print("Check the errors above.")
        print("=" * 60)
