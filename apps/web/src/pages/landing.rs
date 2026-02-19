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
