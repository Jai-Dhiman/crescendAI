# Tutor Knowledge Base Ingestion

This document describes how to ingest piano pedagogy documents into the CrescendAI Tutor system.

## Overview

The ingestion system processes text documents with optional PDFs, chunks them for optimal retrieval, generates embeddings using Cloudflare Workers AI, and stores them in Vectorize for semantic search.

## Architecture

```
Document Files → CLI Parser → Worker API → Embeddings → Vectorize
                                      → Chunks → KV Storage
                                      → PDFs → R2 Storage
                                      → Manifest → R2 Logs
```

## CLI Usage

The ingestion CLI is located at `tools/ingest-kb/`.

### Basic Usage

```bash
cd tools/ingest-kb
cargo run -- --kb-dir ./piano_pedagogy --endpoint https://api.crescend.ai
```

### CLI Options

```bash
ingest-kb [OPTIONS]

Options:
    --kb-dir <PATH>         Directory containing .txt documents [default: ./piano_pedagogy]
    --endpoint <URL>        Worker endpoint URL [required]
    --api-key <KEY>         API key for authentication
    --chunk-size <SIZE>     Target characters per chunk [default: 1000]
    --chunk-overlap <RATIO> Overlap ratio between chunks [default: 0.2]
    --batch-size <SIZE>     Documents per batch [default: 10]
    --validate-only         Only validate setup, don't ingest
    --purge-doc <ID>        Purge specific document by ID
```

### Environment Variables

```bash
export CRESCENDAI_API_KEY="your-api-key"
export CRESCENDAI_ENDPOINT="https://api.crescend.ai"
```

## Document Format

### Text Files

Documents should be `.txt` files with optional header metadata:

```
id: scales_practice_guide
title: Scale Practice Fundamentals
tags: scales, practice, technique, beginner
source: piano_pedagogy
url: https://example.com/scales-guide

This is the body content of the document.
It will be chunked and embedded for retrieval.

Multiple paragraphs are supported...
```

### Header Fields

- `id`: Unique document identifier (required)
- `title`: Human-readable title (optional, defaults to filename)
- `tags`: Comma-separated list of tags (optional)
- `source`: Source category (optional, defaults to "piano_pedagogy")
- `url`: External URL reference (optional)

### Sidecar Metadata

Optional `.meta.json` files can supplement text files:

```json
{
  "title": "Advanced Scale Techniques",
  "tags": ["scales", "advanced", "technique"],
  "source": "masterclass_series",
  "url": "https://example.com/advanced-scales",
  "pdf_filename": "advanced-scales.pdf"
}
```

### PDF Attachments

Place PDF files alongside text files with matching names:

```
scales_guide.txt
scales_guide.pdf
scales_guide.meta.json
```

## Worker API Endpoints

### POST /api/v1/tutor/ingest

Main ingestion endpoint.

**Request Body:**

```json
{
  "documents": [
    {
      "id": "scales_guide",
      "title": "Scale Practice Guide",
      "content": "Complete text content...",
      "tags": ["scales", "practice"],
      "source": "piano_pedagogy",
      "url": "https://example.com/scales",
      "pdf_data": "base64-encoded-pdf-content",
      "pdf_filename": "scales_guide.pdf"
    }
  ],
  "chunking_config": {
    "target_chars": 1000,
    "overlap_ratio": 0.2
  }
}
```

**Response:**

```json
{
  "success": true,
  "documents_processed": 1,
  "chunks_created": 15,
  "errors": [],
  "manifest_path": "ingest/manifests/20241016-143022-uuid.ndjson"
}
```

### POST /api/v1/tutor/ingest/validate

Validate ingestion setup and bindings.

**Response:**

```json
{
  "ai_binding": {"status": "ok"},
  "vectorize_binding": {"status": "ok"},
  "kv_binding": {"status": "ok"},
  "r2_binding": {"status": "ok"},
  "embedding_test": {"status": "ok", "dimensions": 768}
}
```

### POST /api/v1/tutor/ingest/purge

Remove all data for a document.

**Request Body:**

```json
{
  "doc_id": "scales_guide"
}
```

**Response:**

```json
{
  "deleted_vectors": 15,
  "deleted_kv_keys": 15,
  "pdf_cleanup": "manual_required"
}
```

## Chunking Strategy

The system uses character-based chunking with overlap:

- **Target Size**: ~1000 characters per chunk (~250 tokens)
- **Overlap**: 20% overlap between adjacent chunks
- **Boundary Respect**: Chunks break on whitespace when possible
- **Metadata**: Each chunk retains full document metadata

## Storage Layout

### Vectorize

- **Index**: `crescendai-piano-pedagogy`
- **Vector ID**: `{doc_id}::c{chunk_index}`
- **Metadata**: Full KBChunk JSON structure
- **Embedding**: 768-dimensional vectors from `@cf/google/embeddinggemma-300m`

### KV Storage

- **Chunk Data**: `doc:{doc_id}:chunk:{chunk_index}`
- **Content**: Full chunk JSON for retrieval
- **TTL**: No expiration (permanent storage)

### R2 Storage

- **PDFs**: `docs/{doc_id}/{filename}.pdf`
- **Manifests**: `ingest/manifests/{timestamp}-{uuid}.ndjson`
- **Bucket**: `crescendai-practice`

## Example Workflow

1. **Prepare Documents**

   ```bash
   mkdir piano_pedagogy
   echo -e "id: test_doc\ntitle: Test Document\n\nThis is a test." > piano_pedagogy/test_doc.txt
   ```

2. **Validate Setup**

   ```bash
   cargo run -- --validate-only --endpoint https://api.crescend.ai
   ```

3. **Ingest Documents**

   ```bash
   cargo run -- --kb-dir piano_pedagogy --endpoint https://api.crescend.ai
   ```

4. **Verify Ingestion**

   ```bash
   curl -X POST https://api.crescend.ai/api/v1/tutor/retrieve \
     -H "X-API-Key: $CRESCENDAI_API_KEY" \
     -d '{"query": "test content", "top_k": 3}'
   ```

## Error Handling

Common ingestion errors and solutions:

- **Binding Not Found**: Ensure Vectorize/KV/R2 bindings are configured in `wrangler.toml`
- **Embedding Failed**: Check AI binding and model availability
- **Parse Error**: Verify document header format and JSON validity
- **Size Limit**: Split large documents or reduce chunk size
- **Rate Limiting**: Add delays between batch requests

## Performance

- **Throughput**: ~10-50 documents per minute (depends on size)
- **Embedding**: ~200ms per chunk
- **Storage**: ~50ms per chunk (Vectorize + KV)
- **Batch Processing**: Recommended for large document sets

## Monitoring

The system logs structured information:

- Document processing times
- Chunk creation counts
- Storage operation results
- Error details with context

Use the validation endpoint regularly to ensure system health.
