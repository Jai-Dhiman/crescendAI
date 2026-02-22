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
                    <img
                        src="/Image1.jpg"
                        alt="Grand piano seen from above"
                        class="w-full object-cover"
                        style="aspect-ratio: 7/8;"
                    />
                </div>
            </div>
        </section>
    }
}

// -- Social Proof Bar -------------------------------------------------------

#[component]
fn SocialProofBar() -> impl IntoView {
    view! {
        <section class="py-10 md:py-14">
            <div class="container-editorial">
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-8 text-center">
                    <div>
                        <p class="font-display text-display-md text-ink-900">"55%"</p>
                        <p class="text-label-sm uppercase tracking-wider text-clay-600 mt-1">"More accurate"</p>
                    </div>
                    <div>
                        <p class="font-display text-display-md text-ink-900">"30+"</p>
                        <p class="text-label-sm uppercase tracking-wider text-clay-600 mt-1">"Educator interviews"</p>
                    </div>
                    <div>
                        <p class="font-display text-display-md text-ink-900">"15s"</p>
                        <p class="text-label-sm uppercase tracking-wider text-clay-600 mt-1">"To get feedback"</p>
                    </div>
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
                <div class="max-w-2xl mx-auto text-center">
                    <h2 class="font-display text-display-sm md:text-display-lg text-ink-900 mb-6">
                        "Any app can check your notes."
                    </h2>
                    <p class="text-body-lg text-ink-600">
                        "But that's not what separates good playing from great playing. Your tone. Your dynamics. Your phrasing. That's always needed a teacher."
                    </p>
                </div>
            </div>
        </section>
    }
}

// -- How It Works -----------------------------------------------------------

#[component]
fn HowItWorksSection() -> impl IntoView {
    view! {
        <section id="how-it-works" class="scroll-mt-20 py-16 md:py-24">
            <div class="container-editorial">
                <div class="grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-8 lg:gap-16 items-start">
                    // Left column: image
                    <div>
                        <img
                            src="/Image2.jpg"
                            alt="Sheet music resting on vintage piano keys"
                            class="w-full object-cover rounded"
                            style="aspect-ratio: 4/5;"
                        />
                    </div>

                    // Right column: heading + steps
                    <div>
                        <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-4">
                            "How It Works"
                        </h2>
                        <p class="text-body-lg text-ink-600 mb-10">
                            "Under 15 seconds from recording to feedback."
                        </p>

                        <ol class="space-y-8">
                            <li>
                                <span class="font-display text-display-xl text-paper-300 block leading-none">"01"</span>
                                <p class="text-body-md text-ink-700 mt-1">"Record yourself playing"</p>
                            </li>
                            <li>
                                <span class="font-display text-display-xl text-paper-300 block leading-none">"02"</span>
                                <p class="text-body-md text-ink-700 mt-1">"Upload your recording"</p>
                            </li>
                            <li>
                                <span class="font-display text-display-xl text-paper-300 block leading-none">"03"</span>
                                <p class="text-body-md text-ink-700 mt-1">"Get detailed feedback on what to improve"</p>
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
        <section class="py-16 md:py-24">
            <div class="container-editorial">
                <div class="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-8 lg:gap-16 items-start">
                    // Left column: image + feedback cards (first on mobile)
                    <div class="order-2 lg:order-1">
                        <img
                            src="/Image3.jpg"
                            alt="Close-up of classical piano score with dynamic markings"
                            class="w-full object-cover rounded"
                            style="aspect-ratio: 3/2;"
                        />

                        <div class="mt-8 space-y-4">
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

                    // Right column: heading + context (second on mobile)
                    <div class="order-1 lg:order-2">
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
                </div>
            </div>
        </section>
    }
}

// -- Built on Research ------------------------------------------------------

#[component]
fn ResearchSection() -> impl IntoView {
    view! {
        <section class="py-16 md:py-24">
            <div class="container-editorial">
                <div class="grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-8 lg:gap-16 items-start">
                    // Left column: text + stats
                    <div>
                        <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
                            "Built on Research"
                        </h2>
                        <p class="text-body-lg text-ink-600 mb-8">
                            "Trained on thousands of hours of professional performances to hear what a great teacher hears. Published, peer-reviewed, and validated against real educator assessments."
                        </p>

                        <div class="grid grid-cols-2 gap-6 mb-8">
                            <div>
                                <p class="font-display text-display-md text-ink-900">"55%"</p>
                                <p class="text-label-sm uppercase tracking-wider text-clay-600 mt-1">"More accurate"</p>
                            </div>
                            <div>
                                <p class="font-display text-display-md text-ink-900">"30+"</p>
                                <p class="text-label-sm uppercase tracking-wider text-clay-600 mt-1">"Educator interviews"</p>
                            </div>
                        </div>

                        <a
                            href="https://arxiv.org/abs/2601.19029"
                            target="_blank"
                            rel="noopener"
                            class="inline-flex items-center gap-2 text-body-sm text-clay-600 underline underline-offset-2 hover:text-clay-700"
                        >
                            "Published on arXiv"
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                        </a>
                    </div>

                    // Right column: image
                    <div>
                        <img
                            src="/Image4.jpg"
                            alt="Hands playing piano in dramatic light"
                            class="w-full object-cover rounded"
                            style="aspect-ratio: 1/1;"
                        />
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
        <section class="bg-clay-800 text-paper-50 py-20 md:py-32">
            <div class="container-editorial text-center">
                <h2 class="font-display text-display-md md:text-display-xl text-paper-50 mb-8">
                    "Ready to hear what your playing really sounds like?"
                </h2>
                <a href="/analyze" class="btn-primary-inverted text-lg px-8 py-4">
                    "Start Free"
                </a>
            </div>
        </section>
    }
}
