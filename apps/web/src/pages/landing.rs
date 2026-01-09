use leptos::prelude::*;

/// Landing page for the CrescendAI research showcase
/// Academic paper-style layout with abstract intro, methodology, and key findings
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div class="animate-fade-in">
            <HeroSection />
            <MotivationSection />
            <ApproachSection />
            <KeyFindingSection />
            <CtaSection />
            <ApplicationsSection />
        </div>
    }
}

#[component]
fn HeroSection() -> impl IntoView {
    view! {
        <section class="section bg-gradient-paper relative overflow-hidden">
            // Subtle grid pattern background
            <div class="absolute inset-0 bg-grid opacity-50"></div>

            <div class="container-narrow relative text-center">
                <div class="animate-stagger-1">
                    <span class="section-label">
                        "Research Demonstration"
                    </span>
                </div>

                <h1 class="font-display text-display-lg md:text-display-xl lg:text-display-2xl text-ink-900 mb-6 animate-stagger-2">
                    "Perceptual Evaluation of"
                    <span class="block text-sepia-600 mt-2">"Piano Performance"</span>
                </h1>

                <p class="text-body-lg md:text-heading-lg text-ink-500 max-w-2xl mx-auto leading-relaxed mb-10 animate-stagger-3 font-serif">
                    "Exploring how audio-based deep learning models predict human perceptual ratings of piano performances across 19 musical dimensions."
                </p>

                <div class="flex flex-col sm:flex-row items-center justify-center gap-4 animate-stagger-4">
                    <a href="/demo" class="btn-primary text-body-md group">
                        "Explore the Demo"
                        <svg
                            class="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            viewBox="0 0 24 24"
                        >
                            <path stroke-linecap="round" stroke-linejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                        </svg>
                    </a>
                    <a href="#motivation" class="btn-secondary text-body-md">
                        "Read the Research"
                    </a>
                </div>

                // Stats preview
                <div class="mt-16 pt-10 border-t border-paper-300 animate-stagger-5">
                    <div class="grid grid-cols-3 gap-8 max-w-md mx-auto">
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">3</span>
                            <span class="text-label-sm text-ink-400 uppercase tracking-wider">"Models"</span>
                        </div>
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">19</span>
                            <span class="text-label-sm text-ink-400 uppercase tracking-wider">"Dimensions"</span>
                        </div>
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">6</span>
                            <span class="text-label-sm text-ink-400 uppercase tracking-wider">"Recordings"</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn MotivationSection() -> impl IntoView {
    view! {
        <section id="motivation" class="section border-t border-paper-300">
            <div class="container-narrow">
                <span class="section-label">"Motivation"</span>

                <h2 class="font-display text-display-md text-ink-900 mb-8">
                    "Why This Research Matters"
                </h2>

                <div class="prose max-w-none space-y-6">
                    <p class="text-body-lg text-ink-600 leading-relaxed font-serif">
                        "Understanding how humans perceive piano performance quality is essential for music education, automated assessment, and AI-assisted practice tools. Traditional symbolic approaches analyze MIDI data but miss the rich timbral and expressive nuances captured in audio recordings."
                    </p>

                    <p class="text-body-md text-ink-500 leading-relaxed">
                        "This research investigates whether modern audio foundation models can better capture these perceptual qualities directly from the acoustic signal, potentially enabling more accurate and nuanced performance evaluation."
                    </p>
                </div>

                // Research questions
                <div class="mt-12 grid md:grid-cols-2 gap-6">
                    <div class="card p-6">
                        <div class="w-10 h-10 rounded-lg bg-sepia-100 flex items-center justify-center mb-4">
                            <svg class="w-5 h-5 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z"/>
                            </svg>
                        </div>
                        <h3 class="font-display text-heading-lg text-ink-800 mb-2">
                            "Can audio outperform symbolic?"
                        </h3>
                        <p class="text-body-sm text-ink-500">
                            "Do modern audio encoders capture perceptual qualities better than traditional MIDI-based analysis?"
                        </p>
                    </div>

                    <div class="card p-6">
                        <div class="w-10 h-10 rounded-lg bg-sepia-100 flex items-center justify-center mb-4">
                            <svg class="w-5 h-5 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5"/>
                            </svg>
                        </div>
                        <h3 class="font-display text-heading-lg text-ink-800 mb-2">
                            "Which dimensions benefit most?"
                        </h3>
                        <p class="text-body-sm text-ink-500">
                            "How do audio and symbolic modalities complement each other across different perceptual dimensions?"
                        </p>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn ApproachSection() -> impl IntoView {
    view! {
        <section id="approach" class="section bg-paper-200 border-t border-paper-300">
            <div class="container-wide">
                <div class="text-center mb-14">
                    <span class="section-label">"Approach"</span>
                    <h2 class="font-display text-display-md text-ink-900">
                        "Three Model Architectures"
                    </h2>
                </div>

                <div class="grid md:grid-cols-3 gap-8">
                    // Symbolic Model Card
                    <div class="card p-8 hover-lift">
                        <div class="w-14 h-14 rounded-xl bg-paper-300 flex items-center justify-center mb-6">
                            <svg class="w-7 h-7 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z"/>
                            </svg>
                        </div>

                        <h3 class="font-display text-heading-xl text-ink-800 mb-2">
                            "Symbolic"
                        </h3>
                        <p class="text-label-sm text-sepia-500 uppercase tracking-wider mb-4">
                            "PercePiano Baseline"
                        </p>

                        <p class="text-body-sm text-ink-500 mb-6">
                            "Analyzes MIDI representations including note timing, velocity, and pedal events. Captures structural musical features through score alignment."
                        </p>

                        <div class="pt-5 border-t border-paper-300">
                            <span class="text-label-sm text-ink-400">"Published Baseline"</span>
                            <div class="flex items-baseline gap-2 mt-1">
                                <span class="font-mono text-heading-lg text-ink-700">
                                    "R\u{00B2} = 0.397"
                                </span>
                            </div>
                        </div>
                    </div>

                    // Audio Model Card (highlighted)
                    <div class="card p-8 border-sepia-400 bg-sepia-50 shadow-sepia hover-lift relative">
                        // Best badge
                        <div class="absolute -top-3 right-6">
                            <span class="px-3 py-1 text-label-sm font-semibold uppercase tracking-wider bg-sepia-600 text-paper-50 rounded-full shadow-md">
                                "Key Result"
                            </span>
                        </div>

                        <div class="w-14 h-14 rounded-xl bg-sepia-200 flex items-center justify-center mb-6">
                            <svg class="w-7 h-7 text-sepia-700" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"/>
                            </svg>
                        </div>

                        <h3 class="font-display text-heading-xl text-ink-800 mb-2">
                            "Audio"
                        </h3>
                        <p class="text-label-sm text-sepia-600 uppercase tracking-wider mb-4">
                            "MERT-330M"
                        </p>

                        <p class="text-body-sm text-ink-600 mb-6">
                            "Processes raw audio waveforms using a music-specific transformer pretrained on 160K hours. Captures timbre, dynamics, and expression directly."
                        </p>

                        <div class="pt-5 border-t border-sepia-300">
                            <span class="text-label-sm text-sepia-600 flex items-center gap-1">
                                "Best Performance"
                                <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
                                </svg>
                            </span>
                            <div class="flex items-baseline gap-2 mt-1">
                                <span class="font-mono text-heading-lg text-sepia-700">
                                    "R\u{00B2} = 0.433"
                                </span>
                            </div>
                        </div>
                    </div>

                    // Fusion Model Card
                    <div class="card p-8 hover-lift">
                        <div class="w-14 h-14 rounded-xl bg-paper-300 flex items-center justify-center mb-6">
                            <svg class="w-7 h-7 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"/>
                            </svg>
                        </div>

                        <h3 class="font-display text-heading-xl text-ink-800 mb-2">
                            "Fusion"
                        </h3>
                        <p class="text-label-sm text-sepia-500 uppercase tracking-wider mb-4">
                            "Late Fusion"
                        </p>

                        <p class="text-body-sm text-ink-500 mb-6">
                            "Combines symbolic and audio features through learned gating. Leverages complementary strengths of both modalities."
                        </p>

                        <div class="pt-5 border-t border-paper-300">
                            <span class="text-label-sm text-ink-400">"Combined"</span>
                            <div class="flex items-baseline gap-2 mt-1">
                                <span class="font-mono text-heading-lg text-ink-700">
                                    "R\u{00B2} = 0.510"
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn KeyFindingSection() -> impl IntoView {
    view! {
        <section id="finding" class="section">
            <div class="container-narrow">
                <div class="callout">
                    <span class="section-label">"Key Finding"</span>

                    <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
                        "Audio Models Outperform Symbolic Baselines"
                    </h2>

                    <p class="text-body-lg text-ink-600 leading-relaxed mb-8 font-serif">
                        "Our audio-based model achieves "
                        <span class="metric-highlight">"R\u{00B2} = 0.433"</span>
                        ", representing a "
                        <strong class="text-ink-800">"9% relative improvement"</strong>
                        " over the published symbolic baseline ("
                        <span class="font-mono text-ink-500">"R\u{00B2} = 0.397"</span>
                        "). This suggests that perceptual qualities like timbre and expressive nuance are better captured directly from audio than through symbolic representations."
                    </p>

                    <div class="flex flex-wrap items-center gap-8 pt-6 border-t border-sepia-300">
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">"9%"</span>
                            <span class="text-label-sm text-ink-500 uppercase tracking-wider">"Relative Gain"</span>
                        </div>
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">"19"</span>
                            <span class="text-label-sm text-ink-500 uppercase tracking-wider">"Dimensions"</span>
                        </div>
                        <div class="text-center">
                            <span class="block font-mono text-display-sm text-sepia-600">"0.6-0.7"</span>
                            <span class="text-label-sm text-ink-500 uppercase tracking-wider">"Human Ceiling"</span>
                        </div>
                    </div>
                </div>

                // Dimension categories
                <div class="mt-12">
                    <h3 class="font-display text-heading-lg text-ink-800 mb-6">
                        "19 Perceptual Dimensions"
                    </h3>

                    <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div class="p-4 bg-paper-100 rounded-lg border border-paper-200">
                            <h4 class="text-label-md text-sepia-600 uppercase tracking-wider mb-2">"Timing"</h4>
                            <p class="text-body-sm text-ink-500">"timing, articulation length, articulation touch"</p>
                        </div>
                        <div class="p-4 bg-paper-100 rounded-lg border border-paper-200">
                            <h4 class="text-label-md text-sepia-600 uppercase tracking-wider mb-2">"Pedal"</h4>
                            <p class="text-body-sm text-ink-500">"pedal amount, pedal clarity"</p>
                        </div>
                        <div class="p-4 bg-paper-100 rounded-lg border border-paper-200">
                            <h4 class="text-label-md text-sepia-600 uppercase tracking-wider mb-2">"Timbre"</h4>
                            <p class="text-body-sm text-ink-500">"variety, depth, brightness, loudness"</p>
                        </div>
                        <div class="p-4 bg-paper-100 rounded-lg border border-paper-200">
                            <h4 class="text-label-md text-sepia-600 uppercase tracking-wider mb-2">"Expression"</h4>
                            <p class="text-body-sm text-ink-500">"dynamics, tempo, space, balance, drama"</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    }
}

#[component]
fn CtaSection() -> impl IntoView {
    view! {
        <section class="section-sm bg-gradient-cta text-paper-50 relative overflow-hidden">
            // Subtle pattern overlay
            <div class="absolute inset-0 bg-noise opacity-10"></div>

            <div class="container-narrow text-center relative">
                <h2 class="font-display text-display-md text-paper-50 mb-4">
                    "Try the Interactive Demo"
                </h2>

                <p class="text-body-md text-paper-200 mb-10 max-w-lg mx-auto leading-relaxed">
                    "Explore model predictions on recordings by legendary pianists including Horowitz, Argerich, and Gould."
                </p>

                <a
                    href="/demo"
                    class="inline-flex items-center gap-3
                           bg-paper-50 text-sepia-700
                           px-8 py-4 rounded-lg font-medium text-body-md
                           shadow-lg
                           transition-all duration-300
                           hover:bg-paper-100 hover:shadow-xl hover:-translate-y-0.5
                           focus-visible:ring-2 focus-visible:ring-paper-50 focus-visible:ring-offset-2 focus-visible:ring-offset-sepia-700
                           group"
                >
                    "Launch Demo"
                    <svg
                        class="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        viewBox="0 0 24 24"
                    >
                        <path stroke-linecap="round" stroke-linejoin="round" d="M17 8l4 4m0 0l-4 4m4-4H3"/>
                    </svg>
                </a>
            </div>
        </section>
    }
}

#[component]
fn ApplicationsSection() -> impl IntoView {
    view! {
        <section id="applications" class="section border-t border-paper-300">
            <div class="container-wide">
                <div class="text-center mb-14">
                    <span class="section-label">"Applications"</span>
                    <h2 class="font-display text-display-md text-ink-900 mb-4">
                        "Downstream Use Cases"
                    </h2>
                    <p class="text-body-md text-ink-500 max-w-2xl mx-auto">
                        "The perceptual analysis enables practical applications for music education and automated assessment."
                    </p>
                </div>

                <div class="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                    <div class="card p-8 hover-lift">
                        <div class="flex items-center gap-4 mb-5">
                            <div class="w-12 h-12 rounded-xl bg-sepia-100 flex items-center justify-center">
                                <svg class="w-6 h-6 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5"/>
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-lg text-ink-800">
                                "AI Teacher Feedback"
                            </h3>
                        </div>
                        <p class="text-body-md text-ink-500 leading-relaxed">
                            "Generate natural language feedback that identifies strengths and areas for improvement based on the 19-dimension perceptual analysis."
                        </p>
                    </div>

                    <div class="card p-8 hover-lift">
                        <div class="flex items-center gap-4 mb-5">
                            <div class="w-12 h-12 rounded-xl bg-sepia-100 flex items-center justify-center">
                                <svg class="w-6 h-6 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z"/>
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-lg text-ink-800">
                                "Practice Suggestions"
                            </h3>
                        </div>
                        <p class="text-body-md text-ink-500 leading-relaxed">
                            "Derive actionable practice tips targeting specific dimensions that could benefit from focused work, personalized to each performance."
                        </p>
                    </div>
                </div>
            </div>
        </section>
    }
}
