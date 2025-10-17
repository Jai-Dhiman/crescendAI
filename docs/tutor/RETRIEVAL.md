# Tutor Retrieval API

This document describes the CrescendAI Tutor retrieval system API endpoints and usage.

## Overview

The retrieval system provides semantic search over piano pedagogy documents using Cloudflare Workers AI embeddings and Vectorize for vector storage.

## API Endpoints

### POST /api/v1/tutor/retrieve

Embeds a query and returns semantically relevant document chunks.

**Request Body:**
```json
{
  "query": "How to improve timing with metronome practice",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "How to improve timing with metronome practice",
  "results": [
    {
      "id": "metronome_guide::c0",
      "title": "Metronome Practice Fundamentals",
      "source": "piano_pedagogy",
      "text_snippet": "Start metronome practice at a slow tempo, typically 60-80 BPM for scales and basic exercises. Focus on matching each note exactly with the metronome click...",
      "tags": ["timing", "metronome", "practice"],
      "url": "https://example.com/metronome-guide",
      "r2_pdf_key": "docs/metronome_guide/piano_pedagogy.pdf"
    }
  ],
  "total_found": 5
}
```

**Query Limits:**
- Maximum 2KB request body
- Default top_k: 5
- Maximum top_k: 50

### POST /api/v1/tutor

Main tutor feedback endpoint that includes retrieval as part of the ACE pipeline.

**Request Body:**
```json
{
  "analysis": {
    "timing_stable_unstable": 0.3,
    "articulation_short_long": 0.4,
    // ... other 16-dimensional scores
  },
  "user_context": {
    "goals": ["Improve timing", "Better articulation"],
    "practice_time_per_day_minutes": 45,
    "constraints": ["Limited practice space"],
    "repertoire_info": {
      "composer": "Bach",
      "piece": "Invention No. 1",
      "difficulty": 4
    }
  },
  "options": {
    "top_k": 3
  }
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "title": "Practice with metronome at slow tempo",
      "detail": "Focus on: timing_stability",
      "applies_to": ["timing_stability"],
      "practice_plan": ["Start at 60 BPM, increase by 5 BPM when steady"],
      "estimated_time_minutes": 15,
      "citations": ["metronome_guide"]
    }
  ],
  "citations": [
    {
      "id": "metronome_guide",
      "title": "Piano Pedagogy Reference: metronome_guide",
      "source": "Knowledge Base",
      "url": null,
      "sections": ["practice_techniques"]
    }
  ]
}
```

## Authentication

All endpoints require authentication via:
- API Key in `X-API-Key` header
- Rate limiting: 100 requests per minute per IP

## Configuration

The system uses the following environment variables:

- `CF_EMBED_MODEL`: Embedding model (default: `@cf/google/embeddinggemma-300m`)
- `VECTORIZE_INDEX_NAME`: Vectorize index name (default: `crescendai-piano-pedagogy`)
- `TUTOR_TOP_K_DEFAULT`: Default retrieval count (default: `3`)
- `ACE_ENABLED`: Enable ACE pipeline (default: `true`)

## Binding Requirements

The Worker requires these Cloudflare bindings:

- **AI**: `@cf/google/embeddinggemma-300m` for embeddings
- **Vectorize**: `crescendai-piano-pedagogy` index for vector search
- **KV**: `CRESCENDAI_METADATA` for chunk JSON storage
- **R2**: `crescendai-practice` for PDF documents

## Error Handling

The system implements strict error handling with no fallbacks:

**Common Error Responses:**
```json
{
  "error": "Knowledge base query failed: Vectorize binding not found",
  "status": 500
}
```

**Error Categories:**
- `400`: Invalid request (missing query, malformed JSON)
- `401`: Authentication failed
- `429`: Rate limit exceeded
- `500`: System error (misconfiguration, binding failure)

## Citation Format

Retrieved chunks include citations with:
- `id`: Unique chunk identifier
- `title`: Document title
- `source`: Source type (e.g., "piano_pedagogy")
- `text_snippet`: First 200 characters of relevant text
- `tags`: Associated tags for filtering
- `r2_pdf_key`: Path to full PDF in R2 storage

## Performance

- Embedding generation: ~200ms for short queries
- Vectorize query: ~50ms for top-5 results
- Total endpoint latency: ~300-500ms
- Caching: Results cached for 24 hours based on query hash

## Integration with ACE System

The retrieval system integrates with the ACE (Agentic Context Engineering) pipeline:

1. **Generator**: Uses retrieved chunks to inform initial recommendations
2. **Reflector**: Analyzes citation accuracy and relevance
3. **Curator**: Updates playbook based on retrieval effectiveness

## Testing

Use the validation endpoint to test system health:

```bash
curl -X POST https://api.crescend.ai/api/v1/tutor/ingest/validate \
  -H "X-API-Key: YOUR_API_KEY"
```

Expected response includes binding status for AI, Vectorize, KV, and R2.