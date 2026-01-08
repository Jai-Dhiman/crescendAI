use leptos::prelude::*;

use crate::components::PerformanceCard;
#[cfg(not(feature = "ssr"))]
use crate::models::Performance;

#[cfg(feature = "ssr")]
use crate::server_fns::list_performances;

#[component]
pub fn HomePage() -> impl IntoView {
    #[cfg(feature = "ssr")]
    let performances_resource = Resource::new(|| (), |_| list_performances());

    #[cfg(not(feature = "ssr"))]
    let performances_resource: Resource<Result<Vec<Performance>, ServerFnError>> =
        Resource::new(|| (), |_| async { Ok(vec![]) });

    view! {
        <div class="animate-fade-in">
            <section class="section text-center">
                <div class="container-narrow">
                    <div class="flex justify-center mb-8">
                        <div class="flex items-center gap-3">
                            <div class="h-px w-12 bg-gradient-to-r from-transparent to-gold-300"></div>
                            <span class="text-label-md uppercase tracking-widest text-gold-600">
                                "AI-Powered Analysis"
                            </span>
                            <div class="h-px w-12 bg-gradient-to-l from-transparent to-gold-300"></div>
                        </div>
                    </div>

                    <h1 class="text-display-xl md:text-display-2xl font-display font-semibold text-stone-900 mb-6">
                        "Piano Performance"
                        <span class="block text-gradient-gold">"Analysis"</span>
                    </h1>

                    <p class="text-body-lg text-stone-500 max-w-xl mx-auto leading-relaxed mb-10">
                        "Experience detailed feedback on legendary piano performances. "
                        "Discover insights across 19 musical dimensions with our AI-powered system."
                    </p>

                    <div class="flex justify-center gap-6 md:gap-10">
                        <div class="text-center">
                            <span class="block font-display text-display-sm text-gold-600 font-semibold">
                                "19"
                            </span>
                            <span class="text-label-md uppercase tracking-wider text-stone-400">
                                "Dimensions"
                            </span>
                        </div>
                        <div class="w-px bg-stone-200" aria-hidden="true"></div>
                        <div class="text-center">
                            <Suspense fallback=move || view! {
                                <span class="block font-display text-display-sm text-gold-600 font-semibold">
                                    "..."
                                </span>
                            }>
                                {move || {
                                    performances_resource.get().map(|result| {
                                        let count = result.as_ref().map(|p| p.len()).unwrap_or(0);
                                        view! {
                                            <span class="block font-display text-display-sm text-gold-600 font-semibold">
                                                {count}
                                            </span>
                                        }
                                    })
                                }}
                            </Suspense>
                            <span class="text-label-md uppercase tracking-wider text-stone-400">
                                "Performances"
                            </span>
                        </div>
                        <div class="w-px bg-stone-200" aria-hidden="true"></div>
                        <div class="text-center">
                            <span class="block font-display text-display-sm text-gold-600 font-semibold">
                                "AI"
                            </span>
                            <span class="text-label-md uppercase tracking-wider text-stone-400">
                                "Feedback"
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            <div class="accent-line max-w-xs mx-auto mb-12"></div>

            <section class="section-sm" aria-labelledby="gallery-heading">
                <div class="container-wide">
                    <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
                        <div>
                            <h2
                                id="gallery-heading"
                                class="font-display text-display-sm text-stone-900"
                            >
                                "Performance Gallery"
                            </h2>
                            <p class="text-body-sm text-stone-500 mt-1">
                                "Select a recording to receive detailed analysis"
                            </p>
                        </div>
                        <div class="flex items-center gap-2 text-body-sm text-stone-400">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <span>"Click any card to analyze"</span>
                        </div>
                    </div>

                    <Suspense fallback=move || view! {
                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {(0..6).map(|_| view! {
                                <div class="card h-64 animate-pulse bg-stone-100"></div>
                            }).collect_view()}
                        </div>
                    }>
                        {move || {
                            performances_resource.get().map(|result| {
                                match result {
                                    Ok(performances) => view! {
                                        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 stagger">
                                            {performances.iter().cloned().map(|performance| {
                                                view! { <PerformanceCard performance=performance /> }
                                            }).collect_view()}
                                        </div>
                                    }.into_any(),
                                    Err(e) => view! {
                                        <div class="card p-8 text-center">
                                            <div class="w-12 h-12 mx-auto mb-4 rounded-lg bg-error-light flex items-center justify-center">
                                                <svg class="w-6 h-6 text-error" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                                </svg>
                                            </div>
                                            <h3 class="font-display text-heading-lg text-stone-900 mb-2">
                                                "Failed to Load Performances"
                                            </h3>
                                            <p class="text-body-sm text-stone-500">{e.to_string()}</p>
                                        </div>
                                    }.into_any(),
                                }
                            })
                        }}
                    </Suspense>
                </div>
            </section>

            <section id="about" class="section border-t border-stone-200 mt-12" aria-labelledby="about-heading">
                <div class="container-narrow text-center">
                    <div class="flex justify-center mb-6">
                        <span class="badge-gold">"About"</span>
                    </div>

                    <h2
                        id="about-heading"
                        class="font-display text-display-md text-stone-900 mb-6"
                    >
                        "How It Works"
                    </h2>

                    <div class="space-y-4 text-body-md text-stone-600 leading-relaxed">
                        <p>
                            "This demo showcases the PercePiano model, an AI system trained to evaluate "
                            "piano performances across 19 perceptual dimensions including timing, "
                            "articulation, dynamics, and interpretation."
                        </p>
                        <p>
                            "Select any performance from the gallery above to receive detailed AI-powered "
                            "feedback styled as guidance from an encouraging piano teacher."
                        </p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
                        <div class="card p-6 text-left">
                            <div class="w-10 h-10 rounded-md bg-gold-100 flex items-center justify-center mb-4">
                                <svg class="w-5 h-5 text-gold-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-md text-stone-900 mb-2">
                                "19 Dimensions"
                            </h3>
                            <p class="text-body-sm text-stone-500">
                                "Comprehensive analysis across timing, dynamics, articulation, and more."
                            </p>
                        </div>
                        <div class="card p-6 text-left">
                            <div class="w-10 h-10 rounded-md bg-gold-100 flex items-center justify-center mb-4">
                                <svg class="w-5 h-5 text-gold-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-md text-stone-900 mb-2">
                                "AI Feedback"
                            </h3>
                            <p class="text-body-sm text-stone-500">
                                "Personalized guidance styled as encouragement from a piano teacher."
                            </p>
                        </div>
                        <div class="card p-6 text-left">
                            <div class="w-10 h-10 rounded-md bg-gold-100 flex items-center justify-center mb-4">
                                <svg class="w-5 h-5 text-gold-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-md text-stone-900 mb-2">
                                "Practice Tips"
                            </h3>
                            <p class="text-body-sm text-stone-500">
                                "Actionable suggestions to improve your own performances."
                            </p>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    }
}
