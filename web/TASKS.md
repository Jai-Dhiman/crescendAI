# CrescendAI Web Frontend - Implementation Tasks

**Version:** 2.0
**Last Updated:** 2025-11-04
**Status:** Phase 1 - Backend Integration Required

---

## Overview

This document outlines the frontend implementation tasks to match the completed v2.0 backend API. The backend provides comprehensive RAG-powered chat, structured feedback, and recording management capabilities that need to be surfaced in the web interface.

### Current Frontend State

- ✅ **Upload Interface:** File upload with validation and progress tracking
- ✅ **Basic Analysis Display:** Shows model scores and basic visualizations
- ✅ **Component Library:** UI components (Button, Card, Typography)
- ✅ **API Client:** Basic `crescendApi.ts` with upload/analysis endpoints
- ⚠️ **Missing:** Chat interface, structured feedback display, user context management, recording history

### Backend Capabilities (Ready to Use)

From `server/docs/TASKS.md` (Phase 1-7 Complete):
- ✅ **POST /api/v1/upload** - Audio upload with R2 storage
- ✅ **GET /api/v1/recordings** - List recordings with pagination/filtering/sorting
- ✅ **GET /api/v1/recordings/:id** - Get recording metadata
- ✅ **POST /api/v1/feedback/:id** - Generate structured feedback with temporal analysis
- ✅ **POST /api/v1/chat** - Streaming RAG-powered chat (SSE)
- ✅ **POST /api/v1/chat/sessions** - Create chat sessions
- ✅ **GET /api/v1/chat/sessions/:id** - Get session with message history
- ✅ **PUT/GET /api/v1/context** - User context management

---

## Phase 1: API Client Enhancement

**Priority:** P0 (CRITICAL - Required for all features)
**Estimated Effort:** 4 hours
**Dependencies:** None

### Task 1.1: Extend CrescendApiClient

**Description:**
Add missing API methods to `src/lib/services/crescendApi.ts` to match backend v2.0 API.

**Implementation Steps:**

1. Add **Recording Management** methods:
   ```typescript
   // Get single recording with metadata
   async getRecording(recordingId: string): Promise<Recording>

   // List recordings with filters
   async listRecordings(params?: {
     page?: number;
     limit?: number;
     status?: 'uploaded' | 'analyzing' | 'completed';
     date_from?: string;
     date_to?: string;
     sort_by?: 'created_at' | 'updated_at' | 'duration';
     order?: 'asc' | 'desc';
   }): Promise<{ recordings: Recording[]; pagination: Pagination }>
   ```

2. Add **Feedback** method:
   ```typescript
   async getFeedback(recordingId: string): Promise<FeedbackResponse>
   ```

3. Add **Chat** methods:
   ```typescript
   // Create chat session
   async createChatSession(title: string): Promise<ChatSession>

   // Send message with SSE streaming
   async sendChatMessage(
     sessionId: string,
     content: string,
     onToken?: (token: string) => void,
     onToolCall?: (tool: string, args: any) => void
   ): Promise<Message>

   // Get session with history
   async getChatSession(sessionId: string): Promise<ChatSessionWithMessages>

   // List sessions
   async listChatSessions(page?: number, limit?: number): Promise<{
     sessions: ChatSession[];
     pagination: Pagination;
   }>

   // Delete session
   async deleteChatSession(sessionId: string): Promise<void>
   ```

4. Add **User Context** methods:
   ```typescript
   async updateUserContext(context: UserContext): Promise<void>
   async getUserContext(): Promise<UserContext>
   ```

5. Update TypeScript types in `src/lib/types/`:
   - Add `Feedback` types (overall_assessment, temporal_feedback, practice_recommendations)
   - Add `ChatSession` and `Message` types
   - Add `UserContext` type
   - Add `Pagination` type
   - Update `Recording` type to match backend schema

**Acceptance Criteria:**

- [ ] All backend v2.0 endpoints have corresponding client methods
- [ ] TypeScript types match backend API schema exactly
- [ ] SSE streaming works for chat messages
- [ ] Error handling for all methods
- [ ] Methods use existing `makeRequest` helper
- [ ] Input validation before API calls

**Files to Create:**

- `src/lib/types/feedback.ts` (new - feedback types)
- `src/lib/types/chat.ts` (new - chat types)
- `src/lib/types/context.ts` (new - user context types)

**Files to Modify:**

- `src/lib/services/crescendApi.ts` (add ~400 lines)
- `src/lib/types/index.ts` (export new types)

**Status:** ⏳ **PENDING**

---

## Phase 2: Recording Management UI

**Priority:** P0 (HIGH - Core feature)
**Estimated Effort:** 6 hours
**Dependencies:** Task 1.1

### Task 2.1: Recording History Page

**Description:**
Create `/recordings` page to list user's recordings with filtering, sorting, and pagination.

**Implementation Steps:**

1. Create `src/routes/recordings/+page.svelte`:
   - Use `@tanstack/svelte-query` for data fetching
   - Display recordings in table/card view
   - Show: filename, status, duration, created date
   - Add pagination controls (page 1, 2, 3...)
   - Add filter controls (status dropdown, date range picker)
   - Add sort controls (created_at, duration, etc.)
   - Click recording → navigate to `/results/:id`

2. Create `src/lib/components/RecordingCard.svelte`:
   - Display recording metadata
   - Show status badge (uploaded, analyzing, completed)
   - Show duration and file size
   - Action buttons (View Analysis, Delete)

3. Add to navigation in `src/routes/+layout.svelte`:
   - Add "My Recordings" link to nav

**Acceptance Criteria:**

- [ ] Recordings page displays user's recordings
- [ ] Pagination works correctly
- [ ] Filtering by status works
- [ ] Sorting by date/duration works
- [ ] Loading states for data fetching
- [ ] Empty state when no recordings
- [ ] Click recording navigates to analysis page
- [ ] Responsive design (desktop + mobile)

**Files to Create:**

- `src/routes/recordings/+page.svelte` (~250 lines)
- `src/lib/components/RecordingCard.svelte` (~150 lines)

**Files to Modify:**

- `src/routes/+layout.svelte` (add navigation link)

**Status:** ⏳ **PENDING**

---

### Task 2.2: Enhanced Results Page

**Description:**
Update `/results/+page.svelte` to use backend feedback API and display structured feedback.

**Implementation Steps:**

1. Update `src/routes/results/+page.svelte`:
   - Remove mock feedback data
   - Fetch recording metadata from `/api/v1/recordings/:id`
   - Fetch structured feedback from `/api/v1/feedback/:id`
   - Display overall assessment (strengths, areas for improvement, summary)
   - Display temporal feedback timeline
   - Display practice recommendations (immediate + long-term)
   - Display model scores (12D from MERT-330M)
   - Display citations to pedagogy sources

2. Create `src/lib/components/FeedbackDisplay.svelte`:
   - Overall assessment section
   - Strengths/weaknesses cards
   - Summary text

3. Update `src/lib/components/TemporalFeedbackCard.svelte`:
   - Display timestamp-based insights
   - Show score snapshots at key moments
   - Link to audio playback position

4. Update `src/lib/components/PracticeCard.svelte`:
   - Display immediate recommendations
   - Display long-term recommendations
   - Checkable items

5. Create `src/lib/components/CitationsList.svelte`:
   - Display pedagogy sources
   - Show author, source, relevance score
   - Show excerpts

**Acceptance Criteria:**

- [ ] Results page uses real backend feedback API
- [ ] Displays overall assessment clearly
- [ ] Temporal feedback shows timeline visualization
- [ ] Practice recommendations are actionable
- [ ] Citations displayed with relevance scores
- [ ] Loading state while fetching feedback
- [ ] Error handling if feedback generation fails
- [ ] Can regenerate feedback if needed

**Files to Create:**

- `src/lib/components/FeedbackDisplay.svelte` (~200 lines)
- `src/lib/components/CitationsList.svelte` (~100 lines)

**Files to Modify:**

- `src/routes/results/+page.svelte` (refactor ~150 lines)
- `src/lib/components/TemporalFeedbackCard.svelte` (update ~50 lines)
- `src/lib/components/PracticeCard.svelte` (update ~30 lines)

**Status:** ⏳ **PENDING**

---

## Phase 3: RAG-Powered Chat Interface

**Priority:** P1 (HIGH - Key differentiator)
**Estimated Effort:** 10 hours
**Dependencies:** Task 1.1

### Task 3.1: Chat Page

**Description:**
Create `/chat` page with streaming RAG-powered chat interface.

**Implementation Steps:**

1. Create `src/routes/chat/+page.svelte`:
   - Display list of chat sessions (sidebar)
   - Show active chat session messages
   - Input field for new messages
   - "New Session" button
   - Handle SSE streaming for real-time tokens
   - Display tool calls (RAG search indicators)

2. Create `src/lib/components/chat/ChatSidebar.svelte`:
   - List chat sessions
   - Show session title and message count
   - Highlight active session
   - "New Chat" button
   - Delete session button

3. Create `src/lib/components/chat/ChatMessage.svelte`:
   - Display user/assistant messages
   - Show typing indicator
   - Show tool call indicators (e.g., "Searching knowledge base...")
   - Markdown rendering for assistant responses

4. Create `src/lib/components/chat/ChatInput.svelte`:
   - Textarea with auto-resize
   - Send button
   - Character count
   - Loading state during message send

5. Create `src/lib/components/chat/StreamingMessage.svelte`:
   - Display tokens as they arrive via SSE
   - Smooth text appearance animation

**Acceptance Criteria:**

- [ ] Can create new chat sessions
- [ ] Can send messages and receive streaming responses
- [ ] Messages appear in real-time (token by token)
- [ ] Tool calls are visible (e.g., "Searching piano pedagogy...")
- [ ] Session list updates when new session created
- [ ] Can switch between sessions
- [ ] Can delete sessions
- [ ] Message history persists
- [ ] Markdown rendering works
- [ ] Mobile-responsive layout

**Files to Create:**

- `src/routes/chat/+page.svelte` (~400 lines)
- `src/lib/components/chat/ChatSidebar.svelte` (~150 lines)
- `src/lib/components/chat/ChatMessage.svelte` (~100 lines)
- `src/lib/components/chat/ChatInput.svelte` (~120 lines)
- `src/lib/components/chat/StreamingMessage.svelte` (~80 lines)

**Files to Modify:**

- `src/routes/+layout.svelte` (add "Chat" navigation link)

**Status:** ⏳ **PENDING**

---

### Task 3.2: Context-Aware Chat

**Description:**
Integrate user context into chat to personalize responses.

**Implementation Steps:**

1. Create `src/lib/components/chat/ContextPanel.svelte`:
   - Display current user context (goals, constraints, repertoire)
   - "Edit Context" button
   - Shows when context is used in chat

2. Add context indicator to chat messages:
   - Show when assistant uses user context
   - Highlight relevant context details

3. Update chat page to fetch and display context

**Acceptance Criteria:**

- [ ] User context displayed in chat interface
- [ ] Can see when context influences responses
- [ ] Easy access to edit context from chat
- [ ] Context updates reflected in subsequent messages

**Files to Create:**

- `src/lib/components/chat/ContextPanel.svelte` (~120 lines)

**Files to Modify:**

- `src/routes/chat/+page.svelte` (add context panel ~50 lines)

**Status:** ⏳ **PENDING**

---

## Phase 4: User Context Management

**Priority:** P1 (MEDIUM-HIGH)
**Estimated Effort:** 4 hours
**Dependencies:** Task 1.1

### Task 4.1: User Profile/Context Page

**Description:**
Create `/profile` page for managing user context (goals, constraints, repertoire, experience level).

**Implementation Steps:**

1. Create `src/routes/profile/+page.svelte`:
   - Form for editing user context
   - Goals textarea (max 500 chars)
   - Constraints textarea (max 500 chars)
   - Repertoire list (add/remove pieces, max 50)
   - Experience level dropdown (beginner, intermediate, advanced)
   - Save button with loading state

2. Use `@tanstack/svelte-form` for form validation

3. Show character counts for textareas

4. Display saved context on load

**Acceptance Criteria:**

- [ ] Can update goals, constraints, repertoire, experience level
- [ ] Validation works (max lengths, valid values)
- [ ] Character counts displayed
- [ ] Can add/remove repertoire items
- [ ] Saves successfully to backend
- [ ] Shows success/error messages
- [ ] Loading states during save

**Files to Create:**

- `src/routes/profile/+page.svelte` (~300 lines)
- `src/lib/components/RepertoireList.svelte` (~150 lines)

**Files to Modify:**

- `src/routes/+layout.svelte` (add "Profile" navigation link)

**Status:** ⏳ **PENDING**

---

## Phase 5: Enhanced Upload & Workflow

**Priority:** P2 (MEDIUM)
**Estimated Effort:** 3 hours
**Dependencies:** Task 1.1, Task 2.2

### Task 5.1: Upload Flow Enhancement

**Description:**
Improve upload flow to show recording status and generate feedback immediately after upload.

**Implementation Steps:**

1. Update `src/routes/+page.svelte`:
   - After upload completes, show recording ID
   - Show "Generating Feedback" spinner
   - Call `/api/v1/feedback/:id` after upload
   - Navigate to results page when feedback ready

2. Add retry logic for feedback generation (if not ready immediately)

3. Show estimated time for analysis

**Acceptance Criteria:**

- [ ] Upload → Feedback flow is seamless
- [ ] User sees progress through entire workflow
- [ ] Feedback generates automatically after upload
- [ ] Error handling if feedback fails
- [ ] Can navigate to results page after feedback ready

**Files to Modify:**

- `src/routes/+page.svelte` (update workflow ~50 lines)
- `src/lib/services/crescendApi.ts` (add retry logic ~30 lines)

**Status:** ⏳ **PENDING**

---

## Phase 6: Polish & UX Improvements

**Priority:** P2 (MEDIUM)
**Estimated Effort:** 6 hours
**Dependencies:** All previous phases

### Task 6.1: Loading States & Skeletons

**Description:**
Add proper loading states and skeleton screens for all pages.

**Implementation Steps:**

1. Create skeleton components:
   - `RecordingCardSkeleton.svelte`
   - `ChatMessageSkeleton.svelte`
   - `FeedbackSkeleton.svelte`

2. Use in all pages during data fetching

**Acceptance Criteria:**

- [ ] All pages show skeletons while loading
- [ ] Smooth transitions from skeleton to content
- [ ] Loading spinners for actions

**Files to Create:**

- `src/lib/components/skeletons/RecordingCardSkeleton.svelte` (~50 lines)
- `src/lib/components/skeletons/ChatMessageSkeleton.svelte` (~40 lines)
- `src/lib/components/skeletons/FeedbackSkeleton.svelte` (~60 lines)

**Status:** ⏳ **PENDING**

---

### Task 6.2: Error Handling & Retry

**Description:**
Improve error handling with user-friendly messages and retry options.

**Implementation Steps:**

1. Create `src/lib/components/ErrorBoundary.svelte`:
   - Catch errors in component tree
   - Display user-friendly error messages
   - Retry button where applicable

2. Add error states to all API calls

3. Show rate limit errors with retry countdown

**Acceptance Criteria:**

- [ ] All errors have clear user-facing messages
- [ ] Retry buttons where applicable
- [ ] Rate limit errors show countdown
- [ ] Network errors detected and handled

**Files to Create:**

- `src/lib/components/ErrorBoundary.svelte` (~100 lines)

**Files to Modify:**

- All route pages (add error boundaries ~10 lines each)

**Status:** ⏳ **PENDING**

---

### Task 6.3: Mobile Responsiveness

**Description:**
Ensure all pages work well on mobile devices.

**Implementation Steps:**

1. Test all pages on mobile viewports

2. Adjust layouts for mobile:
   - Stack chat sidebar below on mobile
   - Single column layouts
   - Touch-friendly buttons (min 44px)

3. Test upload on mobile devices

**Acceptance Criteria:**

- [ ] All pages responsive on mobile
- [ ] Touch targets large enough
- [ ] Navigation works on mobile
- [ ] Upload works on mobile

**Files to Modify:**

- All route pages (add responsive CSS)

**Status:** ⏳ **PENDING**

---

## Phase 7: Testing & Documentation

**Priority:** P1 (HIGH)
**Estimated Effort:** 6 hours
**Dependencies:** All previous phases

### Task 7.1: Component Tests

**Description:**
Write tests for key components using Vitest and Testing Library.

**Implementation Steps:**

1. Set up Vitest and @testing-library/svelte

2. Write tests for:
   - Upload component
   - Chat components
   - Recording list
   - Feedback display

3. Test user interactions and state changes

**Acceptance Criteria:**

- [ ] Test setup complete
- [ ] Key components have tests
- [ ] >70% test coverage on components
- [ ] CI runs tests on PR

**Files to Create:**

- `src/lib/components/__tests__/` (multiple test files)
- `vitest.config.ts` (new)

**Status:** ⏳ **PENDING**

---

### Task 7.2: E2E Tests

**Description:**
Write end-to-end tests for critical user flows using Playwright.

**Implementation Steps:**

1. Set up Playwright

2. Write E2E tests for:
   - Upload → Results flow
   - Chat session creation and messaging
   - Recording history navigation
   - Profile update

**Acceptance Criteria:**

- [ ] Playwright setup complete
- [ ] Critical flows have E2E tests
- [ ] Tests run in CI
- [ ] Tests use mock backend

**Files to Create:**

- `tests/e2e/` (multiple test files)
- `playwright.config.ts` (new)

**Status:** ⏳ **PENDING**

---

## Summary

### Progress Overview

| Phase | Status | Tasks | Estimated Effort |
|-------|--------|-------|------------------|
| 1. API Client | ⏳ Pending | 1 | 4 hours |
| 2. Recording Management | ⏳ Pending | 2 | 6 hours |
| 3. Chat Interface | ⏳ Pending | 2 | 10 hours |
| 4. User Context | ⏳ Pending | 1 | 4 hours |
| 5. Upload Enhancement | ⏳ Pending | 1 | 3 hours |
| 6. Polish & UX | ⏳ Pending | 3 | 6 hours |
| 7. Testing | ⏳ Pending | 2 | 6 hours |
| **Total** | - | **12 tasks** | **39 hours** |

### Immediate Next Steps (Priority Order)

1. **Task 1.1: Extend CrescendApiClient** (P0 - Required for everything)
   - Add all v2.0 API methods
   - Update TypeScript types
   - Implement SSE streaming for chat

2. **Task 2.2: Enhanced Results Page** (P0 - Core feature)
   - Connect to feedback API
   - Display structured feedback
   - Show temporal analysis and recommendations

3. **Task 2.1: Recording History Page** (P0 - Core feature)
   - List recordings with pagination
   - Add filtering and sorting
   - Navigate to results

4. **Task 3.1: Chat Page** (P1 - Key differentiator)
   - Build streaming chat interface
   - Session management
   - Display tool calls

5. **Task 4.1: User Profile/Context Page** (P1 - Personalization)
   - Context management form
   - Save to backend

### Key Features to Highlight

Once complete, the frontend will showcase:

1. **Comprehensive Feedback System**
   - Overall assessment with strengths/weaknesses
   - Temporal analysis with timestamp-based insights
   - Practice recommendations (immediate + long-term)
   - Citations to piano pedagogy sources

2. **RAG-Powered Chat**
   - Real-time streaming responses
   - Grounded in piano pedagogy knowledge base
   - Context-aware (uses user goals/constraints)
   - Visible tool calling (shows RAG searches)

3. **Recording Management**
   - Complete history with pagination
   - Status tracking (uploaded → analyzing → completed)
   - Filtering by status and date
   - Sorting by various criteria

4. **User Personalization**
   - Goals and constraints
   - Repertoire tracking
   - Experience level
   - Context used throughout app (chat, feedback)

### Backend API Endpoints (Ready to Use)

All backend endpoints from `server/docs/API_REFERENCE.md` are fully implemented and tested:

- ✅ POST /api/v1/upload
- ✅ GET /api/v1/recordings (with pagination, filtering, sorting)
- ✅ GET /api/v1/recordings/:id
- ✅ POST /api/v1/feedback/:id
- ✅ POST /api/v1/chat/sessions
- ✅ POST /api/v1/chat (SSE streaming)
- ✅ GET /api/v1/chat/sessions/:id
- ✅ GET /api/v1/chat/sessions
- ✅ DELETE /api/v1/chat/sessions/:id
- ✅ PUT /api/v1/context
- ✅ GET /api/v1/context
- ✅ GET /health
- ✅ GET /api/status

### Dependencies

**Required npm packages:**
- ✅ `@tanstack/svelte-query` (data fetching)
- ✅ `@tanstack/svelte-form` (forms)
- ⏳ `marked` or `markdown-it` (markdown rendering in chat)
- ⏳ `date-fns` (date formatting)
- ⏳ `eventsource` (SSE polyfill for older browsers)

**Dev Dependencies:**
- ⏳ `vitest` (unit testing)
- ⏳ `@testing-library/svelte` (component testing)
- ⏳ `playwright` (E2E testing)

---

**Last Updated:** 2025-11-04
**Total Estimated Effort:** ~39 hours (1 week for 1 developer)
**Backend Status:** ✅ Complete (ready for integration)
