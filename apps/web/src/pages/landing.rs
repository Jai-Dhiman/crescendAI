use leptos::prelude::*;

/// Editorial magazine-style landing page for Crescend.
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div>
            <HeroSection />
            <hr class="editorial-rule" />
            <SocialProofBar />
            <hr class="editorial-rule" />
            <ProblemSection />
            <hr class="editorial-rule" />
            <HowItWorksSection />
            <hr class="editorial-rule" />
            <FeedbackSection />
            <hr class="editorial-rule" />
            <ResearchSection />
            <hr class="editorial-rule" />
            <FinalCtaSection />
        </div>
    }
}

// -- Hero -------------------------------------------------------------------

#[component]
fn HeroSection() -> impl IntoView {
    view! {
        <section>
            <div class="editorial-bleed">
                <div class="editorial-bleed-text lg:py-32">
                    <h1 class="font-display text-display-md lg:text-display-lg text-ink-900 mb-6">
                        "A teacher for every pianist."
                    </h1>
                    <p class="text-body-lg text-ink-600 mb-8 max-w-md">
                        "Record yourself playing with your phone and get the feedback a great teacher would give you: how to clean up your pedaling, shape your dynamics, warm your tone."
                    </p>
                    <div>
                        <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                            "Start Free"
                        </a>
                        <p class="text-body-sm text-ink-400 mt-3">
                            "No account required"
                        </p>
                    </div>
                </div>

                <div class="editorial-bleed-image">
                    <div class="editorial-placeholder">
                        "Product UI screenshot: radar chart + feedback"
                    </div>
                </div>
            </div>
        </section>
    }
}

// -- Social Proof Bar -------------------------------------------------------

#[component]
fn SocialProofBar() -> impl IntoView {
    view! {
        <section class="py-6 md:py-8">
            <div class="container-editorial">
                <div class="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-0 text-body-sm text-ink-500">
                    <span>"Published research"</span>
                    <span class="hidden sm:block w-px h-4 bg-paper-300 mx-6" aria-hidden="true"></span>
                    <span>"30+ educator interviews"</span>
                    <span class="hidden sm:block w-px h-4 bg-paper-300 mx-6" aria-hidden="true"></span>
                    <span>"55% more accurate than note-based approaches"</span>
                </div>
            </div>
        </section>
    }
}

// -- Problem Statement ------------------------------------------------------

#[component]
fn ProblemSection() -> impl IntoView {
    view! {
        <section class="py-16 md:py-24">
            <div class="container-editorial">
                <div class="editorial-grid">
                    // Left column: punchy copy
                    <div>
                        <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
                            "Any app can check your notes."
                        </h2>
                        <p class="text-body-lg text-ink-600 max-w-md">
                            "But that's not what separates good playing from great playing. Your tone. Your dynamics. Your phrasing. That's always needed a teacher."
                        </p>
                    </div>

                    // Right column: deliberate whitespace
                    <div></div>
                </div>
            </div>
        </section>
    }
}

// -- How It Works -----------------------------------------------------------

#[component]
fn HowItWorksSection() -> impl IntoView {
    view! {
        <section id="how-it-works" class="scroll-mt-20">
            <div class="editorial-bleed">
                <div class="editorial-bleed-text">
                    <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-4">
                        "How It Works"
                    </h2>
                    <p class="text-body-lg text-ink-600">
                        "Under 15 seconds from recording to feedback."
                    </p>
                </div>

                <div class="editorial-bleed-image">
                    <div class="editorial-placeholder">
                        "Product UI: upload to analysis flow"
                    </div>

                    <div class="px-6 sm:px-8 lg:pr-12 mt-8">
                        <ol class="space-y-4">
                            <li class="flex gap-4 items-baseline">
                                <span class="font-display text-display-sm text-paper-400">"1"</span>
                                <span class="text-body-md text-ink-700">"Record yourself playing"</span>
                            </li>
                            <li class="flex gap-4 items-baseline">
                                <span class="font-display text-display-sm text-paper-400">"2"</span>
                                <span class="text-body-md text-ink-700">"Upload your recording"</span>
                            </li>
                            <li class="flex gap-4 items-baseline">
                                <span class="font-display text-display-sm text-paper-400">"3"</span>
                                <span class="text-body-md text-ink-700">"Get detailed feedback on what to improve"</span>
                            </li>
                        </ol>
                    </div>
                </div>
            </div>
        </section>
    }
}

// -- Real Feedback ----------------------------------------------------------

#[component]
fn FeedbackSection() -> impl IntoView {
    view! {
        <section>
            <div class="editorial-bleed">
                <div class="editorial-bleed-text">
                    <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-3">
                        "Real Feedback"
                    </h2>
                    <p class="text-label-sm uppercase tracking-wider text-clay-600 mb-4">
                        "Chopin -- Ballade No. 1 in G minor"
                    </p>
                    <p class="text-body-md text-ink-600 max-w-sm">
                        "Not 'good job' or a letter grade. Specific, actionable feedback on exactly what to practice."
                    </p>
                </div>

                <div class="editorial-bleed-image">
                    <div class="editorial-placeholder">
                        "Zoomed screenshot of feedback card"
                    </div>

                    <div class="px-6 sm:px-8 lg:pr-12 mt-8">
                        <div class="space-y-4">
                            <div class="border-l-2 border-clay-400 pl-4">
                                <p class="text-label-sm font-medium text-clay-700 mb-1">"Sound Quality"</p>
                                <p class="text-body-sm text-ink-600 leading-relaxed">
                                    "Your dynamic range in measures 24-31 stays mostly at mezzo-forte where Chopin's marking calls for a gradual crescendo to fortissimo. Try exaggerating the build -- start softer, arrive louder."
                                </p>
                            </div>
                            <div class="border-l-2 border-clay-400 pl-4">
                                <p class="text-label-sm font-medium text-clay-700 mb-1">"Technical Control"</p>
                                <p class="text-body-sm text-ink-600 leading-relaxed">
                                    "Pedal changes in the lyrical section are clean, but running passages in bars 56-64 accumulate harmonic blur. Try half-pedaling through the chromatic descent."
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

// -- Built on Research ------------------------------------------------------

#[component]
fn ResearchSection() -> impl IntoView {
    view! {
        <section>
            <div class="editorial-bleed">
                <div class="editorial-bleed-text">
                    <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
                        "Built on Research"
                    </h2>
                    <p class="text-body-lg text-ink-600 mb-8 max-w-md">
                        "Trained on thousands of hours of professional performances to hear what a great teacher hears. Published, peer-reviewed, and validated against real educator assessments."
                    </p>

                    <div class="flex flex-wrap gap-x-6 gap-y-2 text-body-sm text-ink-500">
                        <span>"55% more accurate"</span>
                        <span class="text-paper-400" aria-hidden="true">"|"</span>
                        <span>"30+ educator interviews"</span>
                        <span class="text-paper-400" aria-hidden="true">"|"</span>
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="underline underline-offset-2">
                            "Published on arXiv"
                        </a>
                    </div>
                </div>

                <div class="editorial-bleed-image">
                    <div class="editorial-placeholder">
                        "Photo: hands on piano keys"
                    </div>
                </div>
            </div>
        </section>
    }
}

// -- Final CTA --------------------------------------------------------------

#[component]
fn FinalCtaSection() -> impl IntoView {
    view! {
        <section class="py-16 md:py-24">
            <div class="container-editorial">
                <div class="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8 lg:gap-16 items-center">
                    // Left: label
                    <p class="text-label-sm uppercase tracking-wider text-clay-600">
                        "Free to use"
                    </p>

                    // Right: heading + CTA
                    <div>
                        <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
                            "Ready to hear what your playing really sounds like?"
                        </h2>
                        <a href="/analyze" class="btn-primary text-lg px-8 py-4">
                            "Start Free"
                        </a>
                    </div>
                </div>
            </div>
        </section>
    }
}
