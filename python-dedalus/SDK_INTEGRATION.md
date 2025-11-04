# Dedalus SDK Integration Guide

This document describes how the CrescendAI Python Worker integrates with the official Dedalus SDK.

## Overview

The worker now uses the **official Dedalus Python SDK** (`dedalus-labs`) instead of mock responses.

## Installation

The SDK is listed in `requirements.txt`:

```txt
dedalus-labs
```

Wrangler automatically installs dependencies when you run `wrangler dev`.

## Implementation Details

### SDK Components Used

1. **`AsyncDedalus`** - Async client for Dedalus API
2. **`DedalusRunner`** - Orchestrates agent execution with models

### Request Flow

```python
from dedalus_labs import AsyncDedalus, DedalusRunner

# 1. Initialize client with API key
client = AsyncDedalus(api_key=self.api_key)

# 2. Create runner for agent execution
runner = DedalusRunner(client)

# 3. Execute request
result = await runner.run(
    input=user_message,
    model="openai/gpt-5-mini",
    stream=False
)

# 4. Access the response
final_output = result.final_output
```

### Message Format Conversion

The Dedalus SDK uses a simpler input format than OpenAI's messages array:

**Input to Python Worker (OpenAI format):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a piano tutor..."},
    {"role": "user", "content": "How do I improve tempo?"}
  ],
  "model": "openai/gpt-5-mini"
}
```

**Converted to Dedalus format:**
```python
# Extract latest user message
user_message = "How do I improve tempo?"

# Run with DedalusRunner
result = await runner.run(
    input=user_message,
    model="openai/gpt-5-mini"
)
```

**Output (converted back to OpenAI format):**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "To improve tempo consistency...",
      "tool_calls": null
    },
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

## Supported Models

The SDK supports models from multiple providers. Use the format `provider/model-name`:

### OpenAI
- `openai/gpt-5`
- `openai/gpt-5-mini`
- `openai/gpt-4o`

### Anthropic
- `anthropic/claude-sonnet-4-20250514`
- `anthropic/claude-3-5-sonnet-20241022`
- `anthropic/claude-3-5-haiku-20241022`

### Google
- `google/gemini-2.0-flash-exp`
- `google/gemini-1.5-pro`

### Others
- xAI, Perplexity, DeepSeek, Groq, Cohere, Together AI, Cerebras, Mistral

See [Dedalus documentation](https://docs.dedaluslabs.ai/) for the complete list.

## Tool Calling Support

The current implementation does NOT use tool calling through Dedalus. Instead:

1. The Rust worker handles the tool calling loop
2. This Python worker acts as a simple LLM proxy
3. The Rust worker executes tools (like RAG search) locally

### Alternative: Native Dedalus Tool Calling

If you want Dedalus to handle tool calling:

```python
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the piano pedagogy knowledge base.

    Args:
        query: Search query string
        top_k: Number of results to return

    Returns:
        JSON string with search results
    """
    # Make HTTP request back to Rust worker's search endpoint
    # Or implement search logic here
    pass

result = await runner.run(
    input=user_message,
    model="openai/gpt-5-mini",
    tools=[search_knowledge_base]
)
```

The SDK will:
1. Automatically call the function when needed
2. Pass results back to the model
3. Return the final response

## Error Handling

The implementation includes comprehensive error handling:

```python
try:
    result = await runner.run(...)
    return self._convert_to_openai_format(result)
except Exception as e:
    print(f"Error calling Dedalus API: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    raise
```

Errors are:
1. Logged to console for debugging
2. Returned as 500 errors to the Rust worker
3. Include full traceback for investigation

## Configuration

### API Key

Set via environment variable or secret:

```bash
# Development
export DEDALUS_API_KEY="your-key"

# Production
wrangler secret put DEDALUS_API_KEY
```

### Model Selection

The model is specified in each request:

```json
{
  "model": "openai/gpt-5-mini",
  "messages": [...]
}
```

Default model if not specified: `openai/gpt-5-mini`

## Limitations

1. **Token Counts**: The SDK may not expose token usage, so the response returns zeros
2. **Streaming**: Currently disabled (set to `stream=False`)
3. **Tool Calling**: Not implemented in this worker (handled by Rust worker instead)
4. **Message History**: Only the latest user message is passed to Dedalus

## Testing

Test the integration with curl:

```bash
# Health check
curl http://localhost:8788/health

# Chat request
curl -X POST http://localhost:8788/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is a piano arpeggio?"}
    ],
    "model": "openai/gpt-5-mini"
  }'
```

Expected response:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "An arpeggio is a musical technique...",
      "tool_calls": null
    },
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "model": "openai/gpt-5-mini",
  "created": 1699000000
}
```

## Debugging

Enable detailed logging:

1. Check Wrangler console output for Python print statements
2. Look for "Error calling Dedalus API" messages
3. Check the Dedalus API dashboard for request logs

Common issues:
- Invalid API key: Check `DEDALUS_API_KEY` is set correctly
- Invalid model name: Ensure format is `provider/model-name`
- Network errors: Check Cloudflare Workers can reach Dedalus API

## Resources

- [Dedalus SDK Documentation](https://docs.dedaluslabs.ai/llms-full.txt)
- [Dedalus Python Package](https://pypi.org/project/dedalus-labs/)
- [Cloudflare Python Workers](https://developers.cloudflare.com/workers/languages/python/)
