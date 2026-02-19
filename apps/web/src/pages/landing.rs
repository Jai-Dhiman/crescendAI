use leptos::prelude::*;

/// Conversational landing page for Crescend.
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div>
            <HeroSection />
            <ProblemSection />
            <FeedbackShowcase />
            <HowItWorksStrip />
            <MissionSection />
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
                            "A teacher for every pianist."
                        </h1>
                        <p class="text-body-lg text-ink-600 mb-8 max-w-md">
                            "Record yourself playing. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing."
                        </p>
                        <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                            "Try It Free"
                        </a>
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
                                    class="fill-clay-300/70 hero-waveform-bar"
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
                <div class="w-6 h-6 rounded bg-clay-100 flex items-center justify-center">
                    <svg class="w-3.5 h-3.5 text-clay-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                        <path d=icon_path />
                    </svg>
                </div>
                <span class="text-label-sm font-medium text-ink-800">{name}</span>
            </div>
            <p class="text-body-xs text-ink-500 pl-8">{feedback}</p>
        </div>
    }
}

// -- Problem ----------------------------------------------------------------

#[component]
fn ProblemSection() -> impl IntoView {
    view! {
        <section>
            <div class="container-narrow text-center py-16 md:py-24">
                <p class="font-display text-heading-xl md:text-display-sm text-ink-800 leading-relaxed">
                    "Any app can tell you if you played the right notes. But that's not what separates good playing from great playing."
                </p>
                <p class="text-body-lg text-ink-700 font-medium mt-6 max-w-2xl mx-auto">
                    "Your tone. Your dynamics. Your phrasing. That's always needed a teacher."
                </p>
            </div>
        </section>
    }
}

// -- Feedback Showcase ------------------------------------------------------

#[component]
fn FeedbackShowcase() -> impl IntoView {
    view! {
        <section>
            <div class="container-wide py-16 md:py-24">
                <h2 class="font-display text-display-sm md:text-display-md text-ink-900 text-center mb-12">
                    "Here's what Crescend hears"
                </h2>

                <div class="max-w-2xl mx-auto">
                    <div class="card p-6 md:p-8 bg-paper-50/80 backdrop-blur-sm">
                        <div class="mb-4">
                            <p class="text-label-sm uppercase tracking-wider text-clay-600 mb-1">
                                "Chopin -- Ballade No. 1 in G minor"
                            </p>
                            <p class="text-body-sm text-ink-500">"Performed by Krystian Zimerman"</p>
                        </div>

                        <div class="space-y-4">
                            <FeedbackPoint
                                label="Sound Quality"
                                text="Your dynamic range in measures 24-31 stays mostly at mezzo-forte where Chopin's marking calls for a gradual crescendo to fortissimo. Try exaggerating the build -- start softer, arrive louder."
                            />
                            <FeedbackPoint
                                label="Technical Control"
                                text="Pedal changes in the lyrical section are clean, but running passages in bars 56-64 accumulate harmonic blur. Try half-pedaling through the chromatic descent."
                            />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn FeedbackPoint(label: &'static str, text: &'static str) -> impl IntoView {
    view! {
        <div class="border-l-2 border-clay-400 pl-4">
            <p class="text-label-sm font-medium text-clay-700 mb-1">{label}</p>
            <p class="text-body-sm text-ink-600 leading-relaxed">{text}</p>
        </div>
    }
}

// -- How It Works (compact) -------------------------------------------------

#[component]
fn HowItWorksStrip() -> impl IntoView {
    view! {
        <section id="how-it-works" class="scroll-mt-20">
            <div class="container-wide py-10 md:py-14">
                <div class="flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-12 text-center">
                    <StripStep
                        icon_path="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4M12 15a3 3 0 003-3V6a3 3 0 00-6 0v6a3 3 0 003 3z"
                        text="Record yourself"
                    />
                    <svg class="hidden sm:block w-5 h-5 text-clay-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    <StripStep
                        icon_path="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        text="Upload"
                    />
                    <svg class="hidden sm:block w-5 h-5 text-clay-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    <StripStep
                        icon_path="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        text="Get feedback"
                    />
                </div>
            </div>
        </section>
    }
}

#[component]
fn StripStep(icon_path: &'static str, text: &'static str) -> impl IntoView {
    view! {
        <div class="flex items-center gap-2">
            <div class="w-8 h-8 rounded-lg bg-clay-100 flex items-center justify-center flex-shrink-0">
                <svg class="w-4 h-4 text-clay-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d=icon_path />
                </svg>
            </div>
            <span class="text-body-md font-medium text-ink-700">{text}</span>
        </div>
    }
}

// -- Mission + Credibility --------------------------------------------------

#[component]
fn MissionSection() -> impl IntoView {
    view! {
        <section>
            <div class="container-narrow text-center py-16 md:py-24">
                <p class="font-display text-display-sm text-ink-900 mb-8">
                    "Quality feedback shouldn't cost $200 an hour."
                </p>
                <p class="text-body-lg text-ink-600 max-w-2xl mx-auto mb-12">
                    "We built Crescend on published research so that every pianist can practice smarter -- not just those who can afford weekly lessons."
                </p>

                <div class="flex flex-wrap justify-center gap-x-8 gap-y-3 text-body-sm text-ink-500">
                    <span>"55% more accurate than note-based approaches"</span>
                    <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                    <span>"Informed by 30+ educator interviews"</span>
                    <span class="hidden sm:inline text-paper-400" aria-hidden="true">"|"</span>
                    <span>
                        "Published on "
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="underline underline-offset-2">"arXiv"</a>
                    </span>
                </div>
            </div>
        </section>
    }
}

// -- Final CTA --------------------------------------------------------------

#[component]
fn FinalCtaSection() -> impl IntoView {
    view! {
        <section>
            <div class="container-narrow text-center py-16 md:py-24">
                <h2 class="font-display text-display-md text-ink-900 mb-6">
                    "Ready to hear what your playing really sounds like?"
                </h2>
                <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                    "Try It Free"
                </a>
            </div>
        </section>
    }
}
