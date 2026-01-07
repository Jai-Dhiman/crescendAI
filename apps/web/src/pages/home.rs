use leptos::prelude::*;
use crate::components::PerformanceCard;
use crate::api::get_performances;

#[component]
pub fn HomePage() -> impl IntoView {
    let performances = get_performances();

    view! {
        <div>
            // Hero section
            <section class="text-center py-16 mb-8">
                <h1 class="text-5xl md:text-6xl font-serif font-bold mb-6 bg-gradient-to-r from-white via-white to-white/70 bg-clip-text text-transparent">
                    "Piano Performance Analysis"
                </h1>
                <p class="text-xl text-white/60 max-w-2xl mx-auto leading-relaxed">
                    "Experience AI-powered feedback on legendary piano performances. "
                    "Select a recording to discover insights across 19 musical dimensions."
                </p>
                <div class="mt-8 flex justify-center gap-4">
                    <div class="px-4 py-2 bg-slate-800/50 rounded-lg border border-white/10">
                        <span class="text-2xl font-bold text-rose-400">"19"</span>
                        <span class="text-white/50 text-sm ml-2">"Dimensions"</span>
                    </div>
                    <div class="px-4 py-2 bg-slate-800/50 rounded-lg border border-white/10">
                        <span class="text-2xl font-bold text-rose-400">"6"</span>
                        <span class="text-white/50 text-sm ml-2">"Performances"</span>
                    </div>
                </div>
            </section>

            // Gallery section
            <section>
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-2xl font-serif font-semibold text-white">"Performance Gallery"</h2>
                    <p class="text-white/40 text-sm">"Select a performance to analyze"</p>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {performances.into_iter().map(|performance| {
                        view! { <PerformanceCard performance=performance /> }
                    }).collect_view()}
                </div>
            </section>

            // About section
            <section id="about" class="mt-20 py-16 border-t border-white/10">
                <div class="max-w-3xl mx-auto text-center">
                    <h2 class="text-3xl font-serif font-semibold text-white mb-6">"About This Project"</h2>
                    <p class="text-white/60 leading-relaxed mb-4">
                        "This demo showcases the PercePiano model - an AI system trained to evaluate piano performances "
                        "across 19 perceptual dimensions including timing, articulation, dynamics, and interpretation."
                    </p>
                    <p class="text-white/60 leading-relaxed">
                        "Select any performance from the gallery above to receive detailed AI-powered feedback "
                        "styled as guidance from an encouraging piano teacher."
                    </p>
                </div>
            </section>
        </div>
    }
}
