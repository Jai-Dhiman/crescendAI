use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::hooks::use_params_map;
use crate::components::{
    AudioPlayer, LoadingSpinner, PracticeTips, RadarChart, RadarDataPoint, TeacherFeedback,
};
use crate::models::AnalysisState;
use crate::api::{get_loading_messages, get_performance_by_id, mock_analyze_performance};

#[component]
pub fn PerformancePage() -> impl IntoView {
    let params = use_params_map();

    let performance = Memo::new(move |_| {
        let id = params.read().get("id").unwrap_or_default();
        get_performance_by_id(&id)
    });

    let (analysis_state, set_analysis_state) = signal(AnalysisState::Idle);
    let (loading_message, set_loading_message) = signal(String::new());
    let (loading_progress, set_loading_progress) = signal(0u8);

    let start_analysis = move |_| {
        let perf = performance.get();
        if perf.is_none() {
            return;
        }
        let perf_id = perf.unwrap().id;

        set_analysis_state.set(AnalysisState::Loading {
            message: "Starting analysis...".into(),
            progress: 0,
        });

        // Simulate analysis with animated progress
        spawn_local(async move {
            let messages = get_loading_messages();
            for (i, msg) in messages.iter().enumerate() {
                set_loading_message.set(msg.to_string());
                set_loading_progress.set(((i + 1) * 100 / messages.len()) as u8);

                // Simulate delay
                gloo_timers::future::TimeoutFuture::new(600).await;
            }

            let result = mock_analyze_performance(&perf_id);
            set_analysis_state.set(AnalysisState::Complete(result));
        });
    };

    view! {
        <Show
            when=move || performance.get().is_some()
            fallback=|| view! { <PerformanceNotFound /> }
        >
            {move || {
                let perf = performance.get().unwrap();
                view! {
                    <div class="max-w-5xl mx-auto">
                        // Back link
                        <a
                            href="/"
                            class="inline-flex items-center gap-2 text-white/60 hover:text-white mb-6 transition-colors group"
                        >
                            <svg class="w-4 h-4 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                            </svg>
                            "Back to Gallery"
                        </a>

                        // Performance header
                        <header class="mb-8">
                            <p class="text-rose-400 text-lg mb-2">{perf.composer.clone()}</p>
                            <h1 class="text-4xl font-serif font-bold text-white mb-2">{perf.piece_title.clone()}</h1>
                            <p class="text-xl text-white/70">{perf.performer.clone()}</p>
                            {perf.description.clone().map(|desc| view! {
                                <p class="text-white/50 mt-4 max-w-2xl">{desc}</p>
                            })}
                        </header>

                        // Audio player
                        <div class="mb-8">
                            <AudioPlayer
                                audio_url=perf.audio_url.clone()
                                title=format!("{} - {}", perf.piece_title, perf.performer)
                            />
                        </div>

                        // Analysis section
                        {move || {
                            match analysis_state.get() {
                                AnalysisState::Idle => view! {
                                    <div class="text-center py-12 bg-slate-800/30 rounded-xl border border-white/5">
                                        <div class="w-20 h-20 mx-auto mb-6 rounded-full bg-rose-500/10 flex items-center justify-center">
                                            <svg class="w-10 h-10 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                                    d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                            </svg>
                                        </div>
                                        <h3 class="text-xl font-semibold text-white mb-2">"Ready to Analyze"</h3>
                                        <p class="text-white/50 mb-6 max-w-md mx-auto">
                                            "Click below to analyze this performance across 19 musical dimensions and receive personalized feedback."
                                        </p>
                                        <button
                                            on:click=start_analysis
                                            class="px-8 py-4 bg-rose-500 hover:bg-rose-400 rounded-xl text-lg font-semibold transition-all hover:scale-105 shadow-lg shadow-rose-500/25"
                                        >
                                            "Analyze Performance"
                                        </button>
                                        <p class="text-white/40 mt-4 text-sm">
                                            "Analysis typically takes 5-10 seconds"
                                        </p>
                                    </div>
                                }.into_any(),

                                AnalysisState::Loading { .. } => view! {
                                    <LoadingSpinner
                                        message=loading_message
                                        progress=loading_progress
                                    />
                                }.into_any(),

                                AnalysisState::Complete(result) => {
                                    let radar_data: Vec<RadarDataPoint> = result.dimensions.to_labeled_vec()
                                        .into_iter()
                                        .map(|(label, value)| RadarDataPoint {
                                            label: label.to_string(),
                                            value,
                                        })
                                        .collect();

                                    let radar_signal = Signal::derive(move || radar_data.clone());
                                    let feedback = result.teacher_feedback.clone();
                                    let tips = result.practice_tips.clone();

                                    view! {
                                        <div class="space-y-8">
                                            // Results header
                                            <div class="text-center">
                                                <h2 class="text-2xl font-serif font-semibold text-white mb-2">"Analysis Complete"</h2>
                                                <p class="text-white/50">"Here's what we found in this performance"</p>
                                            </div>

                                            // Radar chart
                                            <div class="bg-slate-800/30 rounded-xl p-8 border border-white/5 flex justify-center">
                                                <RadarChart
                                                    data=radar_signal
                                                    size=450
                                                />
                                            </div>

                                            // Feedback and tips in grid
                                            <div class="grid md:grid-cols-2 gap-6">
                                                <TeacherFeedback feedback=feedback />
                                                <PracticeTips tips=tips />
                                            </div>

                                            // Analyze again button
                                            <div class="text-center pt-4">
                                                <a
                                                    href="/"
                                                    class="inline-flex items-center gap-2 text-rose-400 hover:text-rose-300 transition-colors"
                                                >
                                                    "Analyze another performance"
                                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                                                    </svg>
                                                </a>
                                            </div>
                                        </div>
                                    }.into_any()
                                },

                                AnalysisState::Error(msg) => view! {
                                    <div class="text-center py-12 bg-red-900/20 rounded-xl border border-red-500/20">
                                        <p class="text-red-400 mb-4">"Error: " {msg}</p>
                                        <button
                                            on:click=start_analysis
                                            class="text-rose-400 hover:text-rose-300 hover:underline"
                                        >
                                            "Try Again"
                                        </button>
                                    </div>
                                }.into_any(),
                            }
                        }}
                    </div>
                }
            }}
        </Show>
    }
}

#[component]
fn PerformanceNotFound() -> impl IntoView {
    view! {
        <div class="text-center py-20">
            <div class="w-20 h-20 mx-auto mb-6 rounded-full bg-slate-800 flex items-center justify-center">
                <svg class="w-10 h-10 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <h1 class="text-3xl font-serif font-bold text-white mb-4">"Performance Not Found"</h1>
            <p class="text-white/50 mb-6">"The performance you're looking for doesn't exist."</p>
            <a
                href="/"
                class="inline-flex items-center gap-2 px-6 py-3 bg-rose-500 hover:bg-rose-400 rounded-lg font-semibold transition-colors"
            >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                </svg>
                "Return to Gallery"
            </a>
        </div>
    }
}
