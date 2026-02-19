# Website Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform crescend.ai from an academic research showcase into a product-first consumer website targeting serious piano students.

**Architecture:** Complete rewrite of landing page and demo page. New 4-category aggregation model maps 19 PercePiano dimensions to provisional product categories (Sound Quality, Musical Shaping, Technical Control, Interpretive Choices). CSS-only hero animation showing a mock product preview. Upload-first analyze flow with category card results. Existing backend services (HuggingFace inference, RAG feedback, chat) remain unchanged.

**Tech Stack:** Rust, Leptos 0.7 (WASM + SSR), Tailwind CSS, Cloudflare Workers

**Design Direction:** Premium, restrained, serif-forward. "Well-appointed music conservatory" aesthetic. Warm sepia palette, Cormorant Garamond display type, generous whitespace. The word "AI" must not appear in headlines, CTAs, or primary navigation. Consumer copy uses "Crescend" not "CrescendAI."

**Design Doc:** `apps/docs/website-overhaul-design.md`

---

### Task 1: Update Routes, Redirects, and Meta Tags

**Files:**
- Modify: `apps/web/src/app.rs`
- Modify: `apps/web/src/shell.rs`
- Modify: `apps/web/src/pages/mod.rs`

**Step 1: Update app.rs with new routes and footer**

Replace the entire contents of `apps/web/src/app.rs`:

```rust
use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::hooks::use_params_map;
use leptos_router::path;

use crate::components::Header;
use crate::pages::{AnalyzePage, LandingPage};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/output.css"/>

        <Link rel="icon" type_="image/png" href="/crescendai.png"/>
        <Link rel="preconnect" href="https://fonts.googleapis.com"/>
        <Link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous"/>
        <Link
            href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap"
            rel="stylesheet"
        />

        <Title text="Crescend -- Detailed Piano Feedback in Seconds"/>
        <Meta name="description" content="Upload a piano recording and get detailed, personalized feedback on your sound quality, musical shaping, technique, and interpretation."/>

        <Router>
            <div class="min-h-screen bg-gradient-page flex flex-col texture-paper">
                <Header />
                <main class="flex-1">
                    <Routes fallback=|| view! { <NotFound /> }>
                        <Route path=path!("/") view=LandingPage />
                        <Route path=path!("/analyze") view=AnalyzePage />
                        <Route path=path!("/analyze/:id") view=AnalyzePage />
                        <Route path=path!("/demo") view=DemoRedirect />
                        <Route path=path!("/demo/:id") view=DemoRedirect />
                    </Routes>
                </main>
                <Footer />
            </div>
        </Router>
    }
}

#[component]
fn DemoRedirect() -> impl IntoView {
    let params = use_params_map();

    #[cfg(feature = "hydrate")]
    {
        Effect::new(move |_| {
            if let Some(window) = web_sys::window() {
                let id = params.read().get("id");
                let target = match id {
                    Some(id) => format!("/analyze/{}", id),
                    None => "/analyze".to_string(),
                };
                let _ = window.location().replace(&target);
            }
        });
    }

    #[cfg(not(feature = "hydrate"))]
    {
        let _ = params;
    }

    view! {
        <div class="text-center py-20">
            <p class="text-body-md text-ink-500">"Redirecting..."</p>
        </div>
    }
}

#[component]
fn NotFound() -> impl IntoView {
    view! {
        <div class="container-narrow text-center py-24 animate-fade-in">
            <div class="font-display text-[8rem] font-medium text-paper-300 leading-none mb-4">
                "404"
            </div>
            <h1 class="font-display text-display-md text-ink-900 mb-3">
                "Page Not Found"
            </h1>
            <p class="text-body-md text-ink-500 mb-8">
                "The page you're looking for doesn't exist or has been moved."
            </p>
            <a href="/" class="btn-primary">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                </svg>
                "Return Home"
            </a>
        </div>
    }
}

#[component]
fn Footer() -> impl IntoView {
    view! {
        <footer class="border-t border-paper-300 mt-auto bg-paper-100">
            <div class="container-wide py-10">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    // Logo and name
                    <div class="flex items-center gap-3">
                        <img
                            src="/crescendai.png"
                            alt="Crescend Logo"
                            class="w-8 h-8 rounded-md"
                        />
                        <span class="font-display text-lg font-medium text-ink-800">
                            "Crescend"
                        </span>
                    </div>

                    // Links
                    <div class="flex items-center gap-6 text-body-sm">
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="text-sepia-600 hover:text-sepia-700 underline underline-offset-2">
                            "Paper"
                        </a>
                        <a href="mailto:jai@crescend.ai" class="text-sepia-600 hover:text-sepia-700 underline underline-offset-2">
                            "Contact"
                        </a>
                    </div>

                    // Privacy + Copyright
                    <div class="text-right">
                        <p class="text-body-xs text-ink-500 mb-1">
                            "Your recordings are yours. We don't store or train on your data."
                        </p>
                        <p class="text-label-sm text-ink-400">
                            "2026 Crescend"
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    }
}
```

**Step 2: Fix shell.rs body class**

In `apps/web/src/shell.rs`, change:
```
<body class="bg-cream-50 text-stone-800 min-h-screen antialiased">
```
to:
```
<body class="bg-paper-50 text-ink-700 min-h-screen antialiased">
```

**Step 3: Update pages/mod.rs**

Replace `apps/web/src/pages/mod.rs`:

```rust
mod demo;
mod landing;
mod performance;

pub use demo::AnalyzePage;
pub use landing::*;
pub use performance::*;
```

Note: `demo.rs` keeps its filename but the exported component is renamed to `AnalyzePage` (done in Task 9).

**Step 4: Build to verify**

Run: `cd apps/web && cargo check`

Note: This will fail until Task 9 renames `DemoPage` to `AnalyzePage`. If building incrementally, temporarily keep `pub use demo::*;` and update after Task 9.

**Step 5: Commit**

```bash
git add apps/web/src/app.rs apps/web/src/shell.rs apps/web/src/pages/mod.rs
git commit -m "feat: update routes (/analyze), meta tags, footer, and shell"
```

---

### Task 2: Add Category Aggregation Model (TDD)

**Files:**
- Modify: `apps/web/src/models/analysis.rs`

**Step 1: Write the failing tests**

Add to the end of `apps/web/src/models/analysis.rs`:

```rust
/// Composite score for a product category aggregating multiple PercePiano dimensions.
/// Used for the 4-category feedback display on the analyze page.
#[derive(Clone, Debug)]
pub struct CategoryScore {
    pub name: String,
    pub score: f64,
    pub label: String,
    pub summary: String,
    pub practice_tip: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_dimensions(val: f64) -> PerformanceDimensions {
        PerformanceDimensions {
            timing: val,
            articulation_length: val,
            articulation_touch: val,
            pedal_amount: val,
            pedal_clarity: val,
            timbre_variety: val,
            timbre_depth: val,
            timbre_brightness: val,
            timbre_loudness: val,
            dynamics_range: val,
            tempo: val,
            space: val,
            balance: val,
            drama: val,
            mood_valence: val,
            mood_energy: val,
            mood_imagination: val,
            interpretation_sophistication: val,
            interpretation_overall: val,
        }
    }

    #[test]
    fn test_category_scores_returns_four_categories() {
        let dims = uniform_dimensions(0.5);
        let scores = dims.to_category_scores();
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_category_names() {
        let dims = uniform_dimensions(0.5);
        let scores = dims.to_category_scores();
        assert_eq!(scores[0].name, "Sound Quality");
        assert_eq!(scores[1].name, "Musical Shaping");
        assert_eq!(scores[2].name, "Technical Control");
        assert_eq!(scores[3].name, "Interpretive Choices");
    }

    #[test]
    fn test_uniform_high_scores_labeled_strong() {
        let dims = uniform_dimensions(0.8);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Strong");
        }
    }

    #[test]
    fn test_uniform_mid_scores_labeled_good() {
        let dims = uniform_dimensions(0.55);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Good");
        }
    }

    #[test]
    fn test_uniform_low_scores_labeled_developing() {
        let dims = uniform_dimensions(0.35);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Developing");
        }
    }

    #[test]
    fn test_very_low_scores_labeled_needs_focus() {
        let dims = uniform_dimensions(0.15);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert_eq!(score.label, "Needs focus");
        }
    }

    #[test]
    fn test_summary_not_empty() {
        let dims = uniform_dimensions(0.6);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert!(!score.summary.is_empty());
        }
    }

    #[test]
    fn test_practice_tip_not_empty() {
        let dims = uniform_dimensions(0.4);
        let scores = dims.to_category_scores();
        for score in &scores {
            assert!(!score.practice_tip.is_empty());
        }
    }

    #[test]
    fn test_mixed_dimensions_category_average() {
        let mut dims = uniform_dimensions(0.5);
        // Make sound quality dimensions high
        dims.dynamics_range = 0.9;
        dims.timbre_depth = 0.9;
        dims.timbre_variety = 0.9;
        dims.timbre_loudness = 0.9;
        dims.timbre_brightness = 0.9;
        let scores = dims.to_category_scores();
        assert_eq!(scores[0].label, "Strong"); // Sound Quality should be strong
        assert_eq!(scores[1].label, "Good");   // Musical Shaping still at 0.5
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/web && cargo test --lib -- tests`
Expected: FAIL with `method 'to_category_scores' not found`

**Step 3: Implement the category aggregation**

Add the implementation above the `#[cfg(test)]` block in `apps/web/src/models/analysis.rs`:

```rust
impl PerformanceDimensions {
    /// Aggregate 19 PercePiano dimensions into 4 provisional product categories.
    /// Uses equal weights (interim). Weights will be updated to MLP probing R-squared
    /// values once the teacher-grounded taxonomy work is complete.
    pub fn to_category_scores(&self) -> Vec<CategoryScore> {
        let sq_dims = vec![
            ("dynamic range", self.dynamics_range),
            ("tonal depth", self.timbre_depth),
            ("tonal variety", self.timbre_variety),
            ("projection", self.timbre_loudness),
            ("brightness", self.timbre_brightness),
        ];
        let ms_dims = vec![
            ("rhythmic timing", self.timing),
            ("tempo control", self.tempo),
            ("use of space", self.space),
            ("dramatic arc", self.drama),
        ];
        let tc_dims = vec![
            ("pedal use", self.pedal_amount),
            ("pedal clarity", self.pedal_clarity),
            ("note articulation", self.articulation_length),
            ("touch sensitivity", self.articulation_touch),
        ];
        let ic_dims = vec![
            ("emotional expression", self.mood_valence),
            ("musical energy", self.mood_energy),
            ("creative imagination", self.mood_imagination),
            ("interpretive depth", self.interpretation_sophistication),
            ("overall interpretation", self.interpretation_overall),
        ];

        vec![
            build_category("Sound Quality", &sq_dims),
            build_category("Musical Shaping", &ms_dims),
            build_category("Technical Control", &tc_dims),
            build_category("Interpretive Choices", &ic_dims),
        ]
    }
}

fn build_category(name: &str, dims: &[(&str, f64)]) -> CategoryScore {
    let score = dims.iter().map(|(_, v)| v).sum::<f64>() / dims.len() as f64;
    let label = score_to_label(score);
    let summary = generate_summary(dims);
    let weakest = dims.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let practice_tip = dimension_practice_tip(weakest.0).to_string();

    CategoryScore {
        name: name.to_string(),
        score,
        label: label.to_string(),
        summary,
        practice_tip,
    }
}

fn score_to_label(score: f64) -> &'static str {
    if score >= 0.7 {
        "Strong"
    } else if score >= 0.5 {
        "Good"
    } else if score >= 0.3 {
        "Developing"
    } else {
        "Needs focus"
    }
}

fn generate_summary(dims: &[(&str, f64)]) -> String {
    let best = dims.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let worst = dims.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    if (best.1 - worst.1).abs() < 0.05 {
        let avg = dims.iter().map(|(_, v)| v).sum::<f64>() / dims.len() as f64;
        if avg >= 0.6 {
            "Consistently strong across all areas.".to_string()
        } else if avg >= 0.4 {
            "Showing solid foundations across all areas.".to_string()
        } else {
            "This area has room for growth across the board.".to_string()
        }
    } else {
        format!(
            "Your {} stands out as a strength. {} has the most room to develop.",
            best.0, worst.0
        )
    }
}

fn dimension_practice_tip(dim: &str) -> &'static str {
    match dim {
        "dynamic range" => "Practice the same phrase at five dynamic levels (pp through ff), exaggerating contrasts before dialing back to musical levels.",
        "tonal depth" => "Play slow passages focusing on arm weight into the keys. Listen for full resonance before moving to the next note.",
        "tonal variety" => "Try the same melody with fingertip, flat finger, and arm weight touches. Notice how each changes the tonal color.",
        "projection" => "Voice the melody above accompaniment by giving top notes slightly more weight while lightening inner voices.",
        "brightness" => "Use faster key descent with firm fingertips in upper register passages. Listen for clarity and shimmer.",
        "rhythmic timing" => "Practice with a metronome at half tempo, locking each beat precisely. Gradually increase to performance tempo.",
        "tempo control" => "Record yourself and compare to a reference recording. Note where you rush or drag, then target those passages.",
        "use of space" => "Experiment with slight pauses between phrases. Let the music breathe at phrase endings before continuing.",
        "dramatic arc" => "Map the emotional shape of each section before playing. Mark climax points and plan your dynamic trajectory.",
        "pedal use" => "Practice the passage without pedal first to ensure clean technique, then add pedal gradually, listening for blur.",
        "pedal clarity" => "Try half-pedaling through chromatic passages. Change pedal precisely on harmonic changes, not rhythmically.",
        "note articulation" => "Practice staccato and legato versions of the same passage. Focus on consistent, intentional note releases.",
        "touch sensitivity" => "Play scales with each finger producing equal volume. Strengthen fingers 4 and 5 with targeted exercises.",
        "emotional expression" => "Identify the emotional character of each section and play through focusing solely on conveying that emotion.",
        "musical energy" => "Contrast energy levels between sections. Let active passages drive forward and lyrical passages settle back.",
        "creative imagination" => "Listen to three different recordings of the same piece. Note interpretive choices you find compelling and try them.",
        "interpretive depth" => "Research the composer's markings and historical context. Let this knowledge inform your musical decisions.",
        "overall interpretation" => "Record yourself, listen back without the score, and note what you would change. Then address each point.",
        _ => "Focus on this area in your next practice session, starting slowly and building confidence.",
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/web && cargo test --lib -- tests`
Expected: All 9 tests pass.

**Step 5: Commit**

```bash
git add apps/web/src/models/analysis.rs
git commit -m "feat: add 4-category aggregation model with tests"
```

---

### Task 3: Create Category Card Component

**Files:**
- Create: `apps/web/src/components/category_card.rs`
- Modify: `apps/web/src/components/mod.rs`

**Step 1: Create the category card component**

Create `apps/web/src/components/category_card.rs`:

```rust
use leptos::prelude::*;
use crate::models::analysis::CategoryScore;

#[component]
pub fn CategoryCard(category: CategoryScore) -> impl IntoView {
    let bar_width = format!("{}%", (category.score * 100.0).min(100.0));
    let bar_color = match category.label.as_str() {
        "Strong" => "bg-sepia-500",
        "Good" => "bg-sepia-400",
        "Developing" => "bg-sepia-300",
        _ => "bg-sepia-200",
    };

    view! {
        <div class="card p-6">
            <div class="flex items-start gap-4">
                // Category icon
                <div class="flex-shrink-0 w-10 h-10 rounded-lg bg-sepia-100 flex items-center justify-center text-sepia-600">
                    <CategoryIcon name=category.name.clone() />
                </div>

                // Content
                <div class="flex-1 min-w-0">
                    // Header: name + label
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-display text-heading-sm text-ink-800">
                            {&category.name}
                        </h3>
                        <span class="text-label-sm text-sepia-600 font-medium">
                            {&category.label}
                        </span>
                    </div>

                    // Score bar
                    <div class="h-1.5 bg-paper-200 rounded-full overflow-hidden mb-3">
                        <div
                            class=format!("h-full rounded-full transition-all duration-700 {}", bar_color)
                            style=format!("width: {}", bar_width)
                        />
                    </div>

                    // Summary
                    <p class="text-body-sm text-ink-600 mb-3">
                        {&category.summary}
                    </p>

                    // Practice tip
                    <div class="bg-paper-100 rounded-md p-3 border border-paper-200">
                        <p class="text-label-sm text-sepia-600 mb-1">"Practice tip"</p>
                        <p class="text-body-sm text-ink-600">{&category.practice_tip}</p>
                    </div>
                </div>
            </div>
        </div>
    }
}

#[component]
fn CategoryIcon(#[prop(into)] name: String) -> impl IntoView {
    match name.as_str() {
        "Sound Quality" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M12 6v12M8 8v8M16 8v8M4 10v4M20 10v4"/>
            </svg>
        }.into_any(),
        "Musical Shaping" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M3 12c3-6 6-6 9 0s6 6 9 0"/>
            </svg>
        }.into_any(),
        "Technical Control" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="6" width="20" height="12" rx="1"/>
                <path d="M7 6v7M12 6v7M17 6v7M9.5 6v4M14.5 6v4"/>
            </svg>
        }.into_any(),
        "Interpretive Choices" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M9 18V5l8-3v13"/>
                <circle cx="7" cy="18" r="2"/>
                <circle cx="15" cy="15" r="2"/>
            </svg>
        }.into_any(),
        _ => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
            </svg>
        }.into_any(),
    }
}
```

**Step 2: Register the component in mod.rs**

Add to `apps/web/src/components/mod.rs` after the `pub mod teacher_feedback;` line:

```rust
pub mod category_card;
```

And add the re-export after `pub use teacher_feedback::*;`:

```rust
pub use category_card::*;
```

**Step 3: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add apps/web/src/components/category_card.rs apps/web/src/components/mod.rs
git commit -m "feat: add CategoryCard component for 4-category feedback display"
```

---

### Task 4: Update Header Navigation

**Files:**
- Modify: `apps/web/src/components/header.rs`

**Step 1: Rewrite header**

Replace entire contents of `apps/web/src/components/header.rs`:

```rust
use leptos::prelude::*;

#[component]
pub fn Header() -> impl IntoView {
    view! {
        <header class="sticky top-0 z-50 bg-paper-50/95 backdrop-blur-sm border-b border-paper-300">
            <div class="container-wide">
                <div class="flex items-center justify-between h-16">
                    // Logo and brand
                    <a
                        href="/"
                        class="group flex items-center gap-3 text-ink-800 hover:text-sepia-600 transition-colors duration-200"
                        aria-label="Crescend Home"
                    >
                        <div class="relative w-10 h-10 rounded-md overflow-hidden transition-transform duration-200 group-hover:scale-105">
                            <img
                                src="/crescendai.png"
                                alt=""
                                class="w-full h-full object-cover"
                                aria-hidden="true"
                            />
                        </div>
                        <span class="hidden sm:block font-display text-xl font-medium tracking-tight text-ink-900">
                            "Crescend"
                        </span>
                    </a>

                    // Navigation
                    <nav class="flex items-center gap-1" role="navigation" aria-label="Main navigation">
                        <a
                            href="/#how-it-works"
                            class="nav-link hidden sm:inline-flex"
                        >
                            "How It Works"
                        </a>
                        <a
                            href="/analyze"
                            class="nav-link"
                        >
                            "Analyze"
                        </a>

                        <div class="w-px h-5 bg-paper-300 mx-2" aria-hidden="true"></div>

                        // Paper link (external)
                        <a
                            href="https://arxiv.org/abs/2601.19029"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="inline-flex items-center gap-1.5 px-4 py-2 text-body-sm font-medium text-ink-600 rounded-md
                                   transition-all duration-200
                                   hover:bg-paper-200 hover:text-ink-800
                                   focus-visible:ring-2 focus-visible:ring-sepia-500 focus-visible:ring-offset-2"
                            title="Read the research paper on arXiv"
                        >
                            "Paper"
                            <svg class="w-3.5 h-3.5 text-ink-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                            </svg>
                        </a>
                    </nav>
                </div>
            </div>
        </header>
    }
}
```

**Step 2: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add apps/web/src/components/header.rs
git commit -m "feat: update header nav (How It Works, Analyze, Paper)"
```

---

### Task 5: Update Loading Spinner

**Files:**
- Modify: `apps/web/src/components/loading_spinner.rs`

**Step 1: Fix color references and update styling**

Replace entire contents of `apps/web/src/components/loading_spinner.rs`. The current file uses non-existent `stone-*` and `gold-*` color classes. Fix to use `ink-*` and `sepia-*`:

```rust
use leptos::prelude::*;

#[component]
pub fn LoadingSpinner(
    #[prop(into)] message: Signal<String>,
    #[prop(into)] progress: Signal<u8>,
) -> impl IntoView {
    view! {
        <div
            class="card p-12 flex flex-col items-center justify-center text-center"
            role="status"
            aria-live="polite"
            aria-busy="true"
        >
            <div class="relative w-24 h-24 mb-8">
                <svg class="absolute inset-0 w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                    <circle
                        cx="50"
                        cy="50"
                        r="44"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="6"
                        class="text-paper-300"
                    />
                </svg>
                <svg class="absolute inset-0 w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                    <circle
                        cx="50"
                        cy="50"
                        r="44"
                        fill="none"
                        stroke="url(#loadingGradient)"
                        stroke-width="6"
                        stroke-linecap="round"
                        stroke-dasharray="276.46"
                        stroke-dashoffset=move || {
                            let p = progress.get() as f64;
                            276.46 - (276.46 * p / 100.0)
                        }
                        style="transition: stroke-dashoffset 0.4s cubic-bezier(0.16, 1, 0.3, 1)"
                    />
                    <defs>
                        <linearGradient id="loadingGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#8b7355" />
                            <stop offset="100%" stop-color="#6e5a43" />
                        </linearGradient>
                    </defs>
                </svg>
                <div class="absolute inset-0 flex items-center justify-center">
                    <span class="font-display text-display-sm font-semibold text-ink-900">
                        {move || format!("{}", progress.get())}
                    </span>
                    <span class="text-body-sm text-ink-400 ml-0.5">"%"</span>
                </div>
            </div>

            <p class="text-body-md text-ink-600 max-w-sm mb-4">
                {move || message.get()}
            </p>

            <div class="flex gap-1.5" aria-hidden="true">
                <div class="w-2 h-2 rounded-full bg-sepia-400 animate-bounce" style="animation-delay: 0ms"></div>
                <div class="w-2 h-2 rounded-full bg-sepia-500 animate-bounce" style="animation-delay: 150ms"></div>
                <div class="w-2 h-2 rounded-full bg-sepia-600 animate-bounce" style="animation-delay: 300ms"></div>
            </div>

            <span class="sr-only">
                {move || format!("Loading: {}% complete. {}", progress.get(), message.get())}
            </span>
        </div>
    }
}
```

**Step 2: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add apps/web/src/components/loading_spinner.rs
git commit -m "fix: update loading spinner to use sepia/ink color system"
```

---

### Task 6: Add Hero Animation CSS

**Files:**
- Modify: `apps/web/tailwind.css`

**Step 1: Add hero animation keyframes and classes**

Add the following to the `@layer components` section in `apps/web/tailwind.css`, before the closing `}`:

```css
  /* Hero product preview animation */
  @keyframes waveform-pulse {
    0%, 100% { transform: scaleY(0.3); }
    50% { transform: scaleY(1); }
  }

  .hero-waveform-bar {
    transform-box: fill-box;
    transform-origin: bottom;
    animation: waveform-pulse 1.2s ease-in-out infinite;
  }

  @keyframes hero-card-enter {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .hero-card {
    opacity: 0;
    animation: hero-card-enter 0.6s ease-out forwards;
  }
```

Also update the comment at the top of the file. Change:
```css
/* CRESCENDAI RESEARCH SHOWCASE - Academic Paper Aesthetic */
```
to:
```css
/* CRESCEND - Piano Feedback Platform */
```

**Step 2: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors (CSS changes don't affect Rust compilation, but confirm no syntax errors in the CSS by building the full tailwind output).

**Step 3: Commit**

```bash
git add apps/web/tailwind.css
git commit -m "feat: add hero animation CSS keyframes"
```

---

### Task 7: Rewrite Landing Page

**Files:**
- Modify: `apps/web/src/pages/landing.rs`

This is a complete rewrite. The current file is 605 lines of academic research content. Replace with a product-first landing page.

**Step 1: Replace landing.rs entirely**

Replace entire contents of `apps/web/src/pages/landing.rs`:

```rust
use leptos::prelude::*;

/// Product-first landing page for Crescend.
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div>
            <HeroSection />
            <ProblemSection />
            <HowItWorksSection />
            <WhatYouLearnSection />
            <CredibilitySection />
            <FinalCtaSection />
        </div>
    }
}

// -- Hero -------------------------------------------------------------------

#[component]
fn HeroSection() -> impl IntoView {
    view! {
        <section class="relative overflow-hidden">
            <div class="container-wide py-16 md:py-24 lg:py-32">
                <div class="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
                    // Left: Copy
                    <div class="max-w-xl animate-fade-in">
                        <h1 class="font-display text-display-xl md:text-display-2xl text-ink-900 tracking-tight mb-6">
                            "The feedback between lessons"
                        </h1>
                        <p class="text-body-lg text-ink-600 mb-8 max-w-md">
                            "Upload a recording. Get detailed, personalized feedback on your sound, musical shaping, technique, and interpretation."
                        </p>
                        <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                            "Analyze Your Playing"
                        </a>

                        // Social proof strip
                        <div class="flex flex-wrap items-center gap-x-4 gap-y-2 mt-10 text-body-sm text-ink-500">
                            <span>"Backed by published research"</span>
                            <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                            <span>"55% more accurate than note-based approaches"</span>
                            <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                            <span>"Built by a Berklee-trained musician"</span>
                        </div>
                    </div>

                    // Right: Animated product preview
                    <div class="relative animate-fade-in" style="animation-delay: 200ms; animation-fill-mode: both">
                        <HeroProductPreview />
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn HeroProductPreview() -> impl IntoView {
    // Pre-computed waveform bar heights for a natural-looking pattern
    let bars: Vec<(usize, u32)> = vec![
        (0, 12), (1, 22), (2, 35), (3, 25), (4, 42), (5, 30), (6, 48), (7, 20),
        (8, 38), (9, 52), (10, 18), (11, 40), (12, 28), (13, 45), (14, 15),
        (15, 50), (16, 32), (17, 22), (18, 45), (19, 35), (20, 25), (21, 42),
        (22, 18), (23, 52), (24, 30), (25, 38), (26, 22), (27, 48), (28, 15),
        (29, 45), (30, 35), (31, 28), (32, 42), (33, 20), (34, 50), (35, 32),
    ];

    view! {
        <div class="rounded-xl border border-paper-300 bg-paper-50 shadow-elevation-3 overflow-hidden">
            // Mock browser titlebar
            <div class="flex items-center gap-2 px-4 py-3 border-b border-paper-200 bg-paper-100">
                <div class="flex gap-1.5">
                    <div class="w-3 h-3 rounded-full bg-paper-300"></div>
                    <div class="w-3 h-3 rounded-full bg-paper-300"></div>
                    <div class="w-3 h-3 rounded-full bg-paper-300"></div>
                </div>
                <div class="flex-1 mx-4">
                    <div class="bg-paper-200 rounded-md px-3 py-1 text-body-xs text-ink-400 text-center max-w-[200px] mx-auto">
                        "crescend.ai/analyze"
                    </div>
                </div>
            </div>

            // Content area
            <div class="p-5 space-y-3">
                // Animated waveform
                <div class="bg-paper-100 rounded-lg p-3">
                    <svg viewBox="0 0 288 55" class="w-full" preserveAspectRatio="none">
                        {bars.into_iter().map(|(i, h)| {
                            let x = i as u32 * 8;
                            let clamped = h.min(52);
                            view! {
                                <rect
                                    x=format!("{}", x)
                                    y=format!("{}", 55 - clamped)
                                    width="6"
                                    height=format!("{}", clamped)
                                    rx="1"
                                    class="fill-sepia-300/70 hero-waveform-bar"
                                    style=format!("animation-delay: {}ms", i * 80)
                                />
                            }
                        }).collect_view()}
                    </svg>
                </div>

                // Category feedback cards (mock)
                <div class="space-y-2">
                    <HeroMockCard
                        delay="1.5s"
                        icon_path="M12 6v12M8 8v8M16 8v8M4 10v4M20 10v4"
                        name="Sound Quality"
                        feedback="Strong dynamic range with warm, resonant tone..."
                    />
                    <HeroMockCard
                        delay="2s"
                        icon_path="M3 12c3-6 6-6 9 0s6 6 9 0"
                        name="Musical Shaping"
                        feedback="Natural phrasing arc in the second theme..."
                    />
                    <HeroMockCard
                        delay="2.5s"
                        icon_path="M2 6h20v12H2zM7 6v7M12 6v7M17 6v7"
                        name="Technical Control"
                        feedback="Clean pedal transitions in lyrical passages..."
                    />
                </div>
            </div>
        </div>
    }
}

#[component]
fn HeroMockCard(
    delay: &'static str,
    icon_path: &'static str,
    name: &'static str,
    feedback: &'static str,
) -> impl IntoView {
    view! {
        <div class="hero-card rounded-lg border border-paper-200 p-3" style=format!("animation-delay: {}", delay)>
            <div class="flex items-center gap-2 mb-1">
                <div class="w-6 h-6 rounded bg-sepia-100 flex items-center justify-center">
                    <svg class="w-3.5 h-3.5 text-sepia-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                        <path d=icon_path />
                    </svg>
                </div>
                <span class="text-label-sm font-medium text-ink-800">{name}</span>
            </div>
            <p class="text-body-xs text-ink-500 pl-8">{feedback}</p>
        </div>
    }
}

// -- Problem (PAS Framework) ------------------------------------------------

#[component]
fn ProblemSection() -> impl IntoView {
    view! {
        <section class="section-sm">
            <div class="container-narrow text-center">
                <p class="font-display text-heading-xl md:text-display-sm text-ink-800 leading-relaxed">
                    "You practice for hours. But without a teacher's ear, you don't know what to fix."
                </p>
                <p class="text-body-lg text-ink-600 mt-6 max-w-2xl mx-auto">
                    "Is it your pedaling? Your dynamics? Your phrasing? Existing apps check note accuracy -- but that's not what separates good playing from great playing."
                </p>
                <p class="text-body-lg text-ink-700 font-medium mt-6 max-w-2xl mx-auto">
                    "Crescend listens to the things that matter. Not just the right notes -- but how they sound."
                </p>
            </div>
        </section>
    }
}

// -- How It Works -----------------------------------------------------------

#[component]
fn HowItWorksSection() -> impl IntoView {
    view! {
        <section id="how-it-works" class="section bg-paper-100 scroll-mt-20">
            <div class="container-wide">
                <div class="text-center mb-16">
                    <span class="section-label">"How It Works"</span>
                    <h2 class="font-display text-display-md text-ink-900">"Three steps to better practice"</h2>
                </div>

                <div class="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto stagger">
                    <StepCard
                        number=1
                        title="Record"
                        description="Play your piece and record with any device"
                        icon_path="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4M12 15a3 3 0 003-3V6a3 3 0 00-6 0v6a3 3 0 003 3z"
                    />
                    <StepCard
                        number=2
                        title="Upload"
                        description="Upload your recording in seconds"
                        icon_path="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                    <StepCard
                        number=3
                        title="Get Feedback"
                        description="Receive detailed feedback across four dimensions of your playing"
                        icon_path="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                </div>
            </div>
        </section>
    }
}

#[component]
fn StepCard(
    number: u32,
    title: &'static str,
    description: &'static str,
    icon_path: &'static str,
) -> impl IntoView {
    view! {
        <div class="text-center">
            <div class="w-16 h-16 mx-auto mb-6 rounded-xl bg-sepia-100 flex items-center justify-center">
                <svg class="w-8 h-8 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d=icon_path />
                </svg>
            </div>
            <div class="step-indicator mb-4 mx-auto w-fit">
                <span class="step-number">{number}</span>
                <span class="font-medium">{title}</span>
            </div>
            <p class="text-body-md text-ink-600">{description}</p>
        </div>
    }
}

// -- What You'll Learn (4 Categories) ---------------------------------------

#[component]
fn WhatYouLearnSection() -> impl IntoView {
    view! {
        <section class="section">
            <div class="container-wide">
                <div class="text-center mb-16">
                    <span class="section-label">"What You'll Learn"</span>
                    <h2 class="font-display text-display-md text-ink-900">"Feedback across four dimensions"</h2>
                </div>

                <div class="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto stagger">
                    <CategoryPreviewCard
                        name="Sound Quality"
                        description="How does your playing sound? Dynamics, tone, projection."
                        sample="Your dynamic range in measures 24-31 stays mostly at mezzo-forte where Chopin's marking calls for a gradual crescendo to fortissimo."
                        icon_path="M12 6v12M8 8v8M16 8v8M4 10v4M20 10v4"
                    />
                    <CategoryPreviewCard
                        name="Musical Shaping"
                        description="How do you shape the music? Phrasing, timing, flow."
                        sample="The phrasing through the second theme has a natural arc, but the transition at bar 40 feels rushed. Try lingering on the dominant before resolving."
                        icon_path="M3 12c3-6 6-6 9 0s6 6 9 0"
                    />
                    <CategoryPreviewCard
                        name="Technical Control"
                        description="How clean is your technique? Pedaling, articulation, clarity."
                        sample="Pedal changes in the lyrical section are clean, but running passages in bars 56-64 accumulate harmonic blur. Try half-pedaling through the chromatic descent."
                        icon_path="M2 6h20v12H2zM7 6v7M12 6v7M17 6v7"
                    />
                    <CategoryPreviewCard
                        name="Interpretive Choices"
                        description="What story are you telling? Musical decisions, character, expression."
                        sample="The middle section calls for more dramatic contrast -- Chopin marked it agitato for a reason. Let the left hand drive more urgency."
                        icon_path="M9 18V5l8-3v13"
                    />
                </div>
            </div>
        </section>
    }
}

#[component]
fn CategoryPreviewCard(
    name: &'static str,
    description: &'static str,
    sample: &'static str,
    icon_path: &'static str,
) -> impl IntoView {
    view! {
        <div class="card p-6 hover-lift">
            <div class="flex items-start gap-4">
                <div class="flex-shrink-0 w-10 h-10 rounded-lg bg-sepia-100 flex items-center justify-center">
                    <svg class="w-5 h-5 text-sepia-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                        <path d=icon_path />
                    </svg>
                </div>
                <div>
                    <h3 class="font-display text-heading-sm text-ink-800 mb-1">{name}</h3>
                    <p class="text-body-sm text-ink-500 mb-3">{description}</p>
                    <div class="bg-paper-100 rounded-md p-3 border border-paper-200">
                        <p class="text-body-xs text-ink-600 italic leading-relaxed">
                            {sample}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    }
}

// -- Credibility ------------------------------------------------------------

#[component]
fn CredibilitySection() -> impl IntoView {
    view! {
        <section class="section-sm bg-paper-100">
            <div class="container-narrow">
                <div class="text-center mb-12">
                    <span class="section-label">"Credibility"</span>
                    <h2 class="font-display text-display-sm text-ink-900">"Built on published research"</h2>
                </div>

                <div class="grid sm:grid-cols-3 gap-6 mb-12 stagger">
                    <div class="text-center">
                        <div class="font-display text-display-md text-sepia-700 mb-2">"55%"</div>
                        <p class="text-body-sm text-ink-500">"more accurate than note-based approaches"</p>
                    </div>
                    <div class="text-center">
                        <div class="font-display text-display-md text-sepia-700 mb-2">"30+"</div>
                        <p class="text-body-sm text-ink-500">"educator interviews informed the approach"</p>
                    </div>
                    <div class="text-center">
                        <div class="font-display text-display-md text-sepia-700 mb-2">"2026"</div>
                        <p class="text-body-sm text-ink-500">
                            "Published on "
                            <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="underline underline-offset-2">"arXiv"</a>
                            ", submitted to ISMIR"
                        </p>
                    </div>
                </div>

                <div class="text-center">
                    <p class="text-body-md text-ink-600 max-w-lg mx-auto">
                        "Built by Jai Dhiman -- Berklee-trained musician, pianist since age 8, and the engineer behind the research."
                    </p>
                </div>
            </div>
        </section>
    }
}

// -- Final CTA --------------------------------------------------------------

#[component]
fn FinalCtaSection() -> impl IntoView {
    view! {
        <section class="section">
            <div class="container-narrow text-center">
                <h2 class="font-display text-display-md text-ink-900 mb-6">
                    "Ready to hear what your playing really sounds like?"
                </h2>
                <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                    "Analyze Your Playing"
                </a>
            </div>
        </section>
    }
}
```

**Step 2: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add apps/web/src/pages/landing.rs
git commit -m "feat: rewrite landing page from research showcase to product-first"
```

---

### Task 8: Rewrite Analyze Page

**Files:**
- Modify: `apps/web/src/pages/demo.rs`

This task renames `DemoPage` to `AnalyzePage` and replaces the radar chart results with category cards.

**Step 1: Update imports and rename component**

At the top of `apps/web/src/pages/demo.rs`, change the imports from:

```rust
use crate::components::{
    ChatPanel, CollapsibleRadarChart, LoadingSpinner, PracticeTips, RadarDataPoint, TeacherFeedback,
};
```

to:

```rust
use crate::components::{
    CategoryCard, ChatPanel, LoadingSpinner, TeacherFeedback,
};
```

**Step 2: Rename the component and update messaging**

Change the component declaration from:

```rust
/// Interactive Demo Page - upload your own or use demo recordings
#[component]
pub fn DemoPage() -> impl IntoView {
```

to:

```rust
/// Upload and analyze piano recordings with 4-category feedback
#[component]
pub fn AnalyzePage() -> impl IntoView {
```

**Step 3: Update the "no selection" placeholder text**

In the `AnalyzePage` view, find the fallback when no recording is selected. Change:

```rust
<h2 class="font-display text-heading-xl text-ink-800 mb-2">
    "Select a Recording"
</h2>
<p class="text-body-md text-ink-500">
    "Choose a performance above to analyze and explore"
</p>
```

to:

```rust
<h2 class="font-display text-heading-xl text-ink-800 mb-2">
    "Upload or Select a Recording"
</h2>
<p class="text-body-md text-ink-500">
    "Upload your own recording or choose a demo to hear detailed feedback"
</p>
```

**Step 4: Update the upload section label**

In the upload section, change:

```rust
"Upload Your Recording"
```

to:

```rust
"Upload your recording"
```

**Step 5: Update the demo selector label**

Change:

```rust
"Or Select a Demo Performance"
```

to:

```rust
"Don't have a recording? Try one of these:"
```

**Step 6: Update the AnalysisContent idle state**

In the `AnalysisContent` component, change the idle state messaging from:

```rust
<h3 class="font-display text-heading-lg text-ink-900 mb-2">
    "Ready to Analyze"
</h3>
<p class="text-body-md text-ink-500 mb-6 max-w-md mx-auto">
    "Get AI-powered feedback grounded in piano pedagogy sources"
</p>
```

to:

```rust
<h3 class="font-display text-heading-lg text-ink-900 mb-2">
    "Ready for Feedback"
</h3>
<p class="text-body-md text-ink-500 mb-6 max-w-md mx-auto">
    "Get detailed feedback on your sound, phrasing, technique, and interpretation"
</p>
```

And change the button text from `"Analyze Performance"` to `"Get Feedback"`.

**Step 7: Replace AnalysisResults with category cards**

Replace the entire `AnalysisResults` component with:

```rust
#[component]
fn AnalysisResults(result: AnalysisResult, perf_id: String) -> impl IntoView {
    let categories = result.dimensions.to_category_scores();
    let feedback = result.teacher_feedback.clone();

    view! {
        <div class="space-y-6 animate-fade-in">
            // Category Cards
            <div class="space-y-4">
                {categories.into_iter().enumerate().map(|(i, cat)| {
                    view! {
                        <div
                            class="animate-fade-in-up"
                            style=format!("animation-delay: {}ms; animation-fill-mode: both", i * 100)
                        >
                            <CategoryCard category=cat />
                        </div>
                    }
                }).collect_view()}
            </div>

            // Detailed Analysis (Teacher Feedback with citations)
            <div>
                <h3 class="font-display text-heading-lg text-ink-800 mb-4">"Detailed Analysis"</h3>
                <TeacherFeedback feedback=feedback />
            </div>

            // Chat Panel
            <div>
                <h3 class="font-display text-heading-lg text-ink-800 mb-4">"Have a question about your feedback?"</h3>
                <ChatPanel performance_id=perf_id />
            </div>
        </div>
    }
}
```

**Step 8: Update loading messages**

Change the `get_loading_messages` function from:

```rust
#[cfg(feature = "hydrate")]
fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Extracting MuQ embeddings...",
        "Evaluating timing and articulation...",
        "Processing dynamics and pedal...",
        "Assessing timbre and expression...",
        "Generating pedagogical feedback...",
        "Preparing results...",
    ]
}
```

to:

```rust
#[cfg(feature = "hydrate")]
fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Listening to your recording...",
        "Evaluating your sound quality...",
        "Analyzing musical shaping...",
        "Assessing technique and control...",
        "Preparing your feedback...",
    ]
}
```

**Step 9: Build to verify**

Run: `cd apps/web && cargo check`
Expected: Compiles without errors.

**Step 10: Commit**

```bash
git add apps/web/src/pages/demo.rs
git commit -m "feat: rewrite analyze page with 4-category feedback cards"
```

---

### Task 9: Cleanup

**Files:**
- Inspect: `apps/web/public/figures/` (research diagram PNGs)
- Verify: unused component imports

**Step 1: Remove research figure images**

Check if `apps/web/public/figures/` exists. If it does, remove the directory:

```bash
ls apps/web/public/figures/
rm -r apps/web/public/figures/
```

These were embedded in the old research-paper landing page and are no longer referenced.

**Step 2: Verify no broken imports**

Run: `cd apps/web && cargo check`

If there are warnings about unused imports (e.g., `RadarDataPoint`, `PracticeTips` in other files), remove them.

**Step 3: Commit**

```bash
git add -A apps/web/public/
git add apps/web/src/
git commit -m "chore: remove unused research figures and fix imports"
```

---

### Task 10: Build and Verify

**Step 1: Run full build**

```bash
cd apps/web && cargo build --features ssr
cd apps/web && cargo build --features hydrate --target wasm32-unknown-unknown
```

Both must succeed.

**Step 2: Run tests**

```bash
cd apps/web && cargo test --lib
```

All category aggregation tests from Task 2 must pass.

**Step 3: Verification checklist**

Manually verify (or with `cargo check` + local dev server if available):

1. Landing page renders product messaging (not research paper)
2. No "AI" in headlines or CTAs (grep: `grep -rn '"AI' apps/web/src/pages/landing.rs` should return 0 results)
3. Single primary CTA on landing page points to `/analyze`
4. `/analyze` page allows upload and shows 4-category feedback
5. `/demo` redirects to `/analyze`
6. Header navigation: "How It Works", "Analyze", "Paper"
7. Meta title is "Crescend -- Detailed Piano Feedback in Seconds"
8. Footer has privacy note
9. Research figures removed from public assets

**Step 4: Final commit (if any remaining fixes)**

```bash
git add -A apps/web/
git commit -m "chore: final verification and cleanup"
```

---

## Summary of All Changed Files

| File | Action | Task |
|------|--------|------|
| `apps/web/src/app.rs` | Rewrite (routes, footer, redirect) | 1 |
| `apps/web/src/shell.rs` | Fix body class | 1 |
| `apps/web/src/pages/mod.rs` | Update exports | 1 |
| `apps/web/src/models/analysis.rs` | Add CategoryScore + aggregation | 2 |
| `apps/web/src/components/category_card.rs` | Create new | 3 |
| `apps/web/src/components/mod.rs` | Add category_card module | 3 |
| `apps/web/src/components/header.rs` | Rewrite nav | 4 |
| `apps/web/src/components/loading_spinner.rs` | Fix colors | 5 |
| `apps/web/tailwind.css` | Add hero animation CSS | 6 |
| `apps/web/src/pages/landing.rs` | Complete rewrite | 7 |
| `apps/web/src/pages/demo.rs` | Rewrite (rename + category cards) | 8 |
| `apps/web/public/figures/` | Delete directory | 9 |

## Files NOT Changed (kept as-is)

- `apps/web/src/components/audio_upload.rs`
- `apps/web/src/components/audio_player.rs`
- `apps/web/src/components/chat_panel.rs`
- `apps/web/src/components/chat_input.rs`
- `apps/web/src/components/chat_message.rs`
- `apps/web/src/components/expandable_citation.rs`
- `apps/web/src/components/teacher_feedback.rs`
- `apps/web/src/components/radar_chart.rs` (no longer imported but kept for potential future use)
- `apps/web/src/components/practice_tips.rs` (no longer imported but kept)
- `apps/web/src/components/performance_card.rs` (orphaned, kept)
- `apps/web/src/services/*` (all unchanged)
- `apps/web/src/models/performance.rs` (unchanged)
- `apps/web/src/models/pedagogy.rs` (unchanged)
- `apps/web/src/api/*` (unchanged)
- `apps/web/tailwind.config.js` (unchanged)
