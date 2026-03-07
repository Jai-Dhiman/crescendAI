# Chat Interface Design

**Date:** 2026-03-06
**Status:** Approved

## Overview

Add a conversational chat interface to the web app at crescend.ai. The piano teacher is always present in a single continuous conversation thread. Users ask freeform questions; later, audio analysis results will be injected into the same thread for natural-language feedback.

## Architecture

Direct Anthropic calls from the Rust API worker. Streaming SSE responses. Conversation history in D1.

```
Browser --> POST /api/chat (SSE) --> Rust Worker --> Anthropic API (streaming)
                                         |
                                     D1 (conversations + messages)
```

## Data Model (D1)

```sql
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  student_id TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (student_id) REFERENCES students(id)
);

CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,            -- 'user' | 'assistant'
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

No `system` role stored. System prompt constructed server-side.

## API Endpoints

All authenticated via JWT cookie.

### `POST /api/chat` -- Send message, stream response (SSE)

Request:
```json
{ "conversation_id": "uuid | null", "message": "string" }
```

If `conversation_id` is null, creates a new conversation.

Response (SSE):
```
event: message
data: {"type": "start", "conversation_id": "uuid", "message_id": "uuid"}

event: message
data: {"type": "delta", "text": "token"}

event: message
data: {"type": "done", "message_id": "uuid"}
```

Server stores both user message and complete assistant response in D1 after stream finishes. Title generated asynchronously after first exchange.

### `GET /api/conversations` -- List conversations for sidebar

Response:
```json
{ "conversations": [{ "id": "uuid", "title": "string", "updated_at": "iso8601" }] }
```

Ordered by `updated_at` DESC.

### `GET /api/conversations/:id` -- Load conversation history

Response:
```json
{
  "conversation": { "id": "uuid", "title": "string", "created_at": "iso8601" },
  "messages": [{ "id": "uuid", "role": "user|assistant", "content": "string", "created_at": "iso8601" }]
}
```

### `DELETE /api/conversations/:id` -- Delete conversation

Response: 204 No Content.

## LLM Integration

### Model

Anthropic Sonnet 4.6. No model switching mid-conversation.

### System Prompt

```
You are a piano teacher who knows your student well. You have years of
experience and deep knowledge of piano pedagogy, repertoire, and technique.

You're having a conversation with your student. They might ask about
technique, repertoire, practice strategies, music theory, or anything
related to their piano journey. Sometimes you'll also receive analysis
from their practice recordings, which you'll comment on naturally.

How you speak:
- Specific and grounded: reference exact musical concepts, not generalities
- Natural and warm: you're talking to a student you know
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Conversational: match the length and depth to what they asked
```

### Prompt Caching (prefix-match strategy)

Message array construction, ordered for cache hits:

1. **System prompt** (static, `cache_control` breakpoint) -- cached globally
2. **Student context** (level, goals, baselines, injected as first user message) -- cached per student
3. **Full conversation history** (all prior messages) -- cached incrementally
4. **New user message** -- only uncached tokens

Rules (from Claude Code learnings):
- Static content first, dynamic content last
- Never change model mid-conversation
- Use messages for context updates, not system prompt changes
- Full conversation history sent each time (no truncation)

### Title Generation

After the first assistant response completes, a separate non-streaming call to Sonnet generates a 4-6 word title from the first exchange. Fire-and-forget D1 update.

## Frontend

### Layout (claude.ai reference)

**Sidebar** (already stubbed):
- Collapsed by default, expandable
- "New Chat" at top
- Chat list from `GET /api/conversations`, ordered by most recent
- Each item: title + relative time
- Click loads conversation; delete on hover

**Main chat area**:
- Empty state: greeting + input (current design becomes first-message input)
- User messages right-aligned, assistant left-aligned
- Streaming token-by-token display
- Auto-scroll on new messages
- Markdown rendering for assistant responses

**Input bar**:
- Fixed to bottom
- Enter to send, Shift+Enter for newline
- Disabled while streaming

### Routes

- `/app` -- new conversation (empty state)
- `/app/c/:conversationId` -- existing conversation

### State

- Conversations list: fetched on mount, updated on new conversation
- Messages: fetched on navigation to conversation
- Streaming: local React state accumulating tokens
- No state library needed

## Not in Scope

- Audio recording / inference integration (future -- injected into same chat thread)
- File/image uploads
- Voice input/output
- Exercise recommendations in chat
- Mobile responsive design
- Conversation search
- Message editing or regeneration
