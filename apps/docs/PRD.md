# Piano Performance Feedback - Product Requirements Document

## Overview

A research showcase website demonstrating AI-powered piano performance evaluation. Users select from a curated gallery of famous piano performances, click to analyze, and receive natural language feedback from an encouraging virtual piano teacher.

## Goals

1. **Demonstrate PercePiano Research**: Showcase the PercePiano model's ability to evaluate piano performances across 19 perceptual dimensions
2. **Learn Rust**: Build the entire system in Rust where possible, using this project as a learning vehicle
3. **Minimal Friction UX**: One-click analysis with no user accounts, uploads, or configuration

## Target Audience

- Music technology researchers
- Piano educators curious about AI evaluation
- Students exploring automated feedback systems
- General public interested in AI + music intersection

## User Journey

1. User lands on the homepage
2. User sees a gallery of 6-8 famous piano performances (Horowitz, Argerich, etc.)
3. User clicks on a performance to see details (audio player, score preview)
4. User clicks "Analyze Performance"
5. Loading state displays for 5-15 seconds with progress indication
6. Results appear:
   - Radar chart visualization of 19 performance dimensions
   - Encouraging teacher-style written feedback
   - Contextual insights from piano literature (RAG)

## Core Features

### Performance Gallery

- Curated selection of 6-8 iconic piano performances
- Each entry shows: composer, piece title, performer, thumbnail
- Preloaded audio files and MusicXML scores

### Audio Player

- Waveform visualization
- Play/pause controls
- No seeking required (demo simplicity)

### Analysis Results

- **Radar Chart**: Visual representation of all 19 dimensions grouped by category
  - Timing
  - Articulation (length, touch)
  - Pedal (amount, clarity)
  - Timbre (variety, depth, brightness, loudness)
  - Dynamics (range)
  - Tempo, Space, Balance, Drama
  - Mood (valence, energy, imagination)
  - Interpretation (sophistication, interpretation)
- **Teacher Feedback**: 150-200 word encouraging narrative highlighting strengths and growth areas
- **Practice Insights**: 2-3 relevant tips from piano pedagogy literature

### Loading Experience

- Animated progress indicator
- Estimated time remaining
- Brief explanation of what's happening ("Analyzing articulation patterns...")

## Non-Goals (Explicitly Out of Scope)

- User file uploads (all content preloaded)
- User accounts or authentication
- Saving or sharing results
- Mobile-optimized experience (desktop-first demo)
- Real-time analysis during playback
- Comparison between multiple performances
- PDF score rendering (MusicXML metadata only)

## Success Metrics

- Demo completes successfully in <15 seconds (warm state)
- Feedback is coherent and musically relevant
- Radar chart accurately reflects model predictions
- Zero crashes during typical demo session

## Content Requirements

### Demo Performances (6-8 total)

Selection criteria:

- Diverse composers (Chopin, Beethoven, Bach, Rachmaninoff, Prokofiev)
- Diverse performers (Horowitz, Argerich, Zimerman, Kissin, Gould, Pollini)
- Diverse character (virtuosic, lyrical, intellectual, romantic)
- Public domain or licensed audio available
- MusicXML scores available

### Piano Knowledge Base (for RAG)

- Piano pedagogy texts (technique, practice methods)
- Historical performance practice articles
- Famous pianist interviews and masterclass transcripts
- Competition judging criteria documents

## Feedback Tone Guidelines

The virtual teacher should be:

- **Warm and encouraging**: Celebrate strengths before suggesting improvements
- **Specific**: Reference actual musical elements, not vague praise
- **Actionable**: Provide concrete practice strategies
- **Balanced**: Acknowledge both technical and expressive dimensions
- **Humble**: Frame as "observations" not absolute judgments

Example tone:
> "Your sense of timing shows wonderful musical intuition - the subtle rubato in the second theme creates a natural, breathing quality. To further develop your dynamic palette, try practicing the opening phrase at three different dynamic levels while maintaining the same emotional intensity..."

## Constraints

- Budget: <$20/month operational cost
- Latency: <15 seconds end-to-end (acceptable for demo)
- Availability: Best-effort (not production SLA)
- Browser support: Modern Chrome/Firefox/Safari (desktop)
