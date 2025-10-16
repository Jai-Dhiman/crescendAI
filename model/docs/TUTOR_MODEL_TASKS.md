# ACE Framework RAG Tutor Model Implementation

**Status**: Ready to implement - Embedding pipeline is working ✅  
**Timeline**: 2-3 weeks for full implementation  
**Architecture**: 3-agent ACE system (Generator → Reflector → Curator) with local embeddings

---

## 🏗️ System Architecture Overview

```
Piano Recording → Evaluator → Performance Scores → ACE Tutor System → Personalized Feedback
                  (model/)    (16 dimensions)     (web/server)      (actionable insights)

ACE Tutor Components:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  GENERATOR  │───▶│  REFLECTOR  │───▶│   CURATOR   │
│             │    │             │    │             │
│ Produces    │    │ Critiques   │    │ Synthesizes │
│ feedback    │    │ and refines │    │ deltas into │
│ suggestions │    │ insights    │    │ playbook    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                 ┌─────────────────┐
                 │ PIANO PLAYBOOK  │
                 │                 │
                 │ • Practice tips │
                 │ • Technique     │
                 │ • Common issues │
                 │ • Citations     │
                 └─────────────────┘
```

---

## Phase 1: Knowledge Base Setup (Week 1)

### 1.1 Piano Pedagogy Data Collection
**Priority**: High  
**Estimated Time**: 2-3 days

- [ ] **Curate Piano Teaching Materials**
  - Hanon exercises with practice notes
  - Czerny etude techniques 
  - Bach invention fingering guides
  - Chopin etude masterclasses
  - Scale and arpeggio practice methods
  - Common technique problems & solutions

- [ ] **Structure Content for RAG**
  ```
  data/piano-pedagogy/
  ├── technique/
  │   ├── scales-arpeggios.md
  │   ├── finger-independence.md
  │   └── pedaling-techniques.md
  ├── repertoire/
  │   ├── bach-inventions/
  │   ├── chopin-etudes/
  │   └── hanon-exercises/
  ├── common-issues/
  │   ├── timing-problems.md
  │   ├── articulation-issues.md
  │   └── pedal-clarity.md
  └── practice-strategies/
      ├── slow-practice.md
      ├── mental-practice.md
      └── metronome-work.md
  ```

- [ ] **Create Content Chunking Script**
  ```python
  # scripts/chunk_pedagogy_content.py
  - Split documents into semantic chunks (200-400 tokens)
  - Add metadata: difficulty_level, technique_focus, piece_type
  - Generate embeddings using local service
  - Store in Cloudflare Vectorize with metadata
  ```

### 1.2 Embedding & Vector Database Setup
**Priority**: High  
**Estimated Time**: 1-2 days

- [ ] **Integrate Local Embeddings with Knowledge Base**
  - Update `server/src/knowledge_base.rs` with local embedding calls
  - Implement batch processing for efficient ingestion
  - Add fallback to Cloudflare AI if local service unavailable

- [ ] **Populate Vectorize Index**
  ```bash
  # Example workflow
  cd scripts/
  python chunk_pedagogy_content.py --input ../data/piano-pedagogy/ --output chunks.jsonl
  python ingest_to_vectorize.py --chunks chunks.jsonl --index crescendai-piano-pedagogy
  ```

- [ ] **Test Retrieval Quality**
  - Query: "student plays scales with uneven timing"
  - Expected: Metronome practice, slow practice techniques
  - Measure: Semantic similarity, relevance scores

---

## Phase 2: ACE Agent Implementation (Week 2)

### 2.1 Generator Agent
**Priority**: High  
**Estimated Time**: 2-3 days

- [ ] **Implement Initial Feedback Generator**
  ```rust
  // server/src/tutor/generator.rs
  pub struct FeedbackGenerator {
      knowledge_base: Arc<KnowledgeBase>,
      llm_client: Arc<LLMClient>,
  }
  
  impl FeedbackGenerator {
      pub async fn generate_initial_feedback(
          &self,
          scores: &PerformanceScores,
          user_context: &UserContext,
      ) -> Result<InitialFeedback>
  }
  ```

- [ ] **Query Strategy Implementation**
  - Identify weakest 3-4 dimensions from evaluator scores
  - Build retrieval query: "technique focus: timing_stability, repertoire: {piece_info}"
  - Retrieve top-k relevant chunks from knowledge base
  - Generate structured feedback using LLM with citations

- [ ] **Feedback Template System**
  ```json
  {
    "immediate_actions": [
      "Use metronome at 60 BPM for scales",
      "Practice hands separately first"
    ],
    "technique_focus": "timing_stability",
    "practice_duration": "15-20 minutes daily",
    "expected_improvement": "More consistent timing in 1-2 weeks",
    "citations": ["hanon_01::practice_tips", "metronome_work::steady_tempo"]
  }
  ```

### 2.2 Reflector Agent
**Priority**: Medium  
**Estimated Time**: 2-3 days

- [ ] **Feedback Quality Assessment**
  ```rust
  // server/src/tutor/reflector.rs
  pub struct FeedbackReflector {
      critique_prompt: String,
      llm_client: Arc<LLMClient>,
  }
  
  // Analyzes generated feedback for:
  // - Actionability (specific vs. vague)
  // - Citation accuracy (do references support claims?)
  // - User context alignment (skill level appropriate?)
  // - Completeness (missing important aspects?)
  ```

- [ ] **Delta Generation Logic**
  - Extract concrete insights from reflection
  - Identify patterns in successful/unsuccessful feedback
  - Generate improvement deltas for next iteration
  - Track confidence scores for feedback quality

### 2.3 Curator Agent  
**Priority**: Medium  
**Estimated Time**: 2-3 days

- [ ] **Playbook Management System**
  ```rust
  // server/src/tutor/curator.rs
  pub struct TutorCurator {
      playbook: Arc<Mutex<PianoPlaybook>>,
      kv_store: Arc<KvStore>,
  }
  
  #[derive(Serialize, Deserialize)]
  pub struct PlaybookEntry {
      id: String,
      technique: String,
      advice: String,
      success_count: u32,
      failure_count: u32,
      confidence: f32,
      last_updated: DateTime<Utc>,
      citations: Vec<String>,
  }
  ```

- [ ] **Adaptive Learning Mechanism**
  - Track which advice leads to user improvement
  - Increment success/failure counters based on follow-up sessions
  - Prune low-confidence or outdated entries
  - Merge similar entries to avoid redundancy

---

## Phase 3: Integration & Optimization (Week 3)

### 3.1 Full Pipeline Integration
**Priority**: High  
**Estimated Time**: 2-3 days

- [ ] **End-to-End Workflow**
  ```
  1. Audio → Evaluator → 16D Performance Scores
  2. Scores → Generator → Initial Feedback + Citations
  3. Feedback → Reflector → Quality Assessment + Deltas  
  4. Deltas → Curator → Updated Playbook
  5. Return: Structured feedback with citations
  ```

- [ ] **API Endpoint Updates**
  ```rust
  // server/src/handlers.rs
  pub async fn generate_tutor_feedback_ace(
      req: Request, 
      ctx: RouteContext<()>
  ) -> Result<Response> {
      // 1. Parse performance scores
      // 2. Run through ACE pipeline
      // 3. Return enhanced feedback with playbook insights
  }
  ```

### 3.2 Caching & Performance
**Priority**: Medium  
**Estimated Time**: 1-2 days

- [ ] **Multi-Level Caching Strategy**
  ```rust
  // Cache hierarchy:
  // L1: In-memory LRU (recent feedback)
  // L2: KV store (session-based) 
  // L3: Vectorize (knowledge retrieval)
  
  // Cache keys:
  let cache_key = format!("ace_feedback:{}:{}", 
                         hash_performance_scores(scores),
                         hash_user_context(context));
  ```

- [ ] **Intelligent Cache Invalidation**
  - Invalidate when playbook updates significantly
  - Preserve across similar score patterns
  - Implement TTL based on feedback confidence

### 3.3 Quality Assurance & Testing
**Priority**: High  
**Estimated Time**: 2-3 days

- [ ] **ACE Pipeline Tests**
  ```python
  # tests/test_ace_pipeline.py
  def test_generator_produces_actionable_feedback():
      scores = mock_performance_scores(timing_stable_unstable=0.3)
      feedback = generator.generate(scores, user_context)
      assert "metronome" in feedback.immediate_actions[0].lower()
      assert feedback.citations is not empty
  
  def test_reflector_improves_feedback_quality():
      initial = mock_vague_feedback()  
      reflected = reflector.reflect(initial)
      assert reflected.specificity_score > initial.specificity_score
  
  def test_curator_learns_from_patterns():
      # Simulate successful advice pattern
      curator.record_success("metronome_practice", user_improvement=0.8)
      playbook = curator.get_playbook()
      metronome_entry = playbook.get("timing_stability")
      assert metronome_entry.confidence > 0.7
  ```

---

## Phase 4: Advanced Features (Future)

### 4.1 Multi-Modal Enhancements
- [ ] **Score Analysis Integration**
  - Parse MusicXML/MIDI for piece-specific advice
  - Align performance issues with notation challenges
  - Generate fingering suggestions

- [ ] **Video Analysis** (Advanced)
  - Posture and hand position feedback
  - Visual technique assessment
  - Combine with audio analysis

### 4.2 Personalization Engine
- [ ] **Learning Style Adaptation**
  - Visual vs. auditory learners
  - Technical vs. musical focus
  - Practice time constraints

- [ ] **Progress Tracking**
  - Long-term improvement trends
  - Skill development milestones
  - Adaptive difficulty adjustment

---

## 📊 Success Metrics

### Technical Metrics
- [ ] **Response Time**: < 5 seconds for complete ACE cycle
- [ ] **Cache Hit Rate**: > 70% for similar score patterns
- [ ] **Knowledge Retrieval**: > 0.8 semantic similarity for relevant results
- [ ] **Citation Accuracy**: 100% of citations must link to valid sources

### Quality Metrics  
- [ ] **Specificity**: Advice should be actionable within 24-48 hours
- [ ] **Relevance**: > 80% of suggestions should address top 3 weakest dimensions
- [ ] **Improvement Correlation**: Track if students who follow advice show measurable improvement

### User Experience Metrics
- [ ] **Comprehensiveness**: Cover technique, practice strategy, and musical aspects
- [ ] **Encouragement**: Include positive reinforcement and realistic timelines
- [ ] **Accessibility**: Appropriate for stated skill level and practice time

---

## 🛠️ Implementation Commands

```bash
# Phase 1: Knowledge Base
cd scripts/
python collect_piano_pedagogy.py
python chunk_and_embed.py --local-embedding-service http://localhost:8001
python ingest_vectorize.py --test-retrieval

# Phase 2: ACE Agents  
cd server/
cargo build --release
cargo test tutor::ace_pipeline

# Phase 3: Integration
cd web/
bun run build
cd ../server/
wrangler deploy

# Phase 4: Testing
cd tests/
python test_ace_end_to_end.py --performance-scores test_data/sample_scores.json
```

---

## 🔗 Dependencies

- ✅ Local embedding service (completed)
- ⏳ Cloudflare Vectorize index setup
- ⏳ Piano pedagogy content collection
- ⏳ LLM integration (Cloudflare AI or OpenAI)
- ⏳ KV store for playbook persistence

**Ready to start Phase 1! The embedding foundation is solid.**