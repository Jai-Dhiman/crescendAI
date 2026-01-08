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
                    <h1 class="text-display-xl md:text-display-2xl font-display font-semibold text-stone-900 mb-6">
                        "Piano Performance"
                        <span class="block text-gradient-gold">"Analysis"</span>
                    </h1>

                    <p class="text-body-lg text-stone-500 max-w-xl mx-auto leading-relaxed">
                        "A research demo exploring AI evaluation of piano performances "
                        "across 19 perceptual dimensions."
                    </p>
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
                    <h2
                        id="about-heading"
                        class="font-display text-display-sm text-stone-900 mb-6"
                    >
                        "About This Research"
                    </h2>

                    <p class="text-body-md text-stone-600 leading-relaxed max-w-2xl mx-auto mb-10">
                        "This demo compares three model architectures for piano performance evaluation. "
                        "Select a performance from the gallery to see AI-generated analysis and feedback."
                    </p>

                    <div class="flex flex-wrap justify-center gap-4 md:gap-8">
                        <div class="text-center px-4">
                            <span class="block font-mono text-body-sm font-medium text-stone-800">"Symbolic"</span>
                            <span class="text-label-sm text-stone-400">"PercePiano"</span>
                        </div>
                        <div class="hidden md:block w-px h-8 bg-stone-200"></div>
                        <div class="text-center px-4">
                            <span class="block font-mono text-body-sm font-medium text-stone-800">"Audio"</span>
                            <span class="text-label-sm text-stone-400">"MERT"</span>
                        </div>
                        <div class="hidden md:block w-px h-8 bg-stone-200"></div>
                        <div class="text-center px-4">
                            <span class="block font-mono text-body-sm font-medium text-gold-600">"Fusion"</span>
                            <span class="text-label-sm text-stone-400">"Combined"</span>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    }
}
