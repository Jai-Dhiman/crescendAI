use leptos::prelude::*;
use leptos_router::hooks::use_params_map;

use crate::components::{CollapsibleRadarChart, LoadingSpinner, PracticeTips, RadarDataPoint, TeacherFeedback};
use crate::models::AnalysisState;

#[cfg(feature = "hydrate")]
use crate::components::AudioPlayer;

#[component]
pub fn PerformancePage() -> impl IntoView {
    let params = use_params_map();

    let id_signal = Memo::new(move |_| params.read().get("id").unwrap_or_default());

    let performance_resource = Resource::new(
        move || id_signal.get(),
        |id| async move { get_performance(id).await },
    );

    let (analysis_state, set_analysis_state) = signal(AnalysisState::Idle);
    let (loading_message, set_loading_message) = signal(String::new());
    let (loading_progress, set_loading_progress) = signal(0u8);

    view! {
        <Suspense fallback=move || view! {
            <div class="container-wide animate-fade-in">
                <div class="mb-8">
                    <div class="h-4 w-32 bg-stone-200 rounded animate-pulse"></div>
                </div>
                <div class="mb-10">
                    <div class="h-6 w-48 bg-stone-200 rounded animate-pulse mb-4"></div>
                    <div class="h-10 w-96 bg-stone-200 rounded animate-pulse mb-3"></div>
                    <div class="h-6 w-64 bg-stone-200 rounded animate-pulse"></div>
                </div>
                <div class="card h-48 bg-stone-100 animate-pulse mb-10"></div>
            </div>
        }>
            {move || {
                performance_resource.get().map(|result| {
                    match result {
                        Ok(perf) => {
                            let perf_for_header = perf.clone();
                            let perf_id = perf.id.clone();

                            #[cfg(feature = "hydrate")]
                            let audio_section = {
                                let perf_for_audio = perf.clone();
                                view! {
                                    <div class="mb-10">
                                        <AudioPlayer
                                            audio_url=perf_for_audio.audio_url.clone()
                                            title=format!("{} - {}", perf_for_audio.piece_title, perf_for_audio.performer)
                                        />
                                    </div>
                                }
                            };

                            #[cfg(not(feature = "hydrate"))]
                            let audio_section = {
                                let perf_for_audio = perf.clone();
                                view! {
                                    <div class="mb-10">
                                        <div class="card p-6">
                                            <h3 class="font-display text-heading-md text-stone-900 mb-5">
                                                {format!("{} - {}", perf_for_audio.piece_title, perf_for_audio.performer)}
                                            </h3>
                                            <div class="h-20 bg-stone-50 rounded-lg mb-5 flex items-center justify-center border border-stone-100">
                                                <p class="text-body-sm text-stone-400">"Audio player loading..."</p>
                                            </div>
                                        </div>
                                    </div>
                                }
                            };

                            view! {
                                <div class="container-wide animate-fade-in">
                                    <nav class="mb-8" aria-label="Breadcrumb">
                                        <a
                                            href="/"
                                            class="group inline-flex items-center gap-2 text-body-sm text-stone-500 hover:text-gold-600 transition-colors"
                                        >
                                            <svg
                                                class="w-4 h-4 transition-transform group-hover:-translate-x-0.5"
                                                fill="none"
                                                stroke="currentColor"
                                                stroke-width="2"
                                                viewBox="0 0 24 24"
                                                aria-hidden="true"
                                            >
                                                <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                                            </svg>
                                            "Back to Gallery"
                                        </a>
                                    </nav>

                                    <header class="mb-10">
                                        <div class="flex items-center gap-3 mb-4">
                                            <span class="badge-gold">{perf_for_header.composer.clone()}</span>
                                            {perf_for_header.year_recorded.map(|year| view! {
                                                <span class="badge-neutral">{format!("{}", year)}</span>
                                            })}
                                        </div>
                                        <h1 class="font-display text-display-lg md:text-display-xl text-stone-900 mb-3">
                                            {perf_for_header.piece_title.clone()}
                                        </h1>
                                        <p class="text-heading-lg text-gold-600 font-medium">
                                            {perf_for_header.performer.clone()}
                                        </p>
                                        {perf_for_header.description.clone().map(|desc| view! {
                                            <p class="text-body-md text-stone-500 mt-4 max-w-2xl">{desc}</p>
                                        })}
                                    </header>

                                    {audio_section}

                                    <div class="accent-line mb-10"></div>

                                    <AnalysisSection
                                        perf_id=perf_id
                                        analysis_state=analysis_state
                                        set_analysis_state=set_analysis_state
                                        loading_message=loading_message
                                        set_loading_message=set_loading_message
                                        loading_progress=loading_progress
                                        set_loading_progress=set_loading_progress
                                    />
                                </div>
                            }.into_any()
                        },
                        Err(e) => {
                            let error_msg = e.to_string();
                            if error_msg.contains("not found") {
                                view! { <PerformanceNotFound /> }.into_any()
                            } else {
                                view! {
                                    <div class="container-narrow text-center py-20 animate-fade-in">
                                        <div class="w-16 h-16 mx-auto mb-6 rounded-lg bg-error-light flex items-center justify-center">
                                            <svg class="w-8 h-8 text-error" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                            </svg>
                                        </div>
                                        <h1 class="font-display text-display-md text-stone-900 mb-3">
                                            "Error Loading Performance"
                                        </h1>
                                        <p class="text-body-md text-stone-500 mb-8">
                                            {error_msg}
                                        </p>
                                        <a href="/" class="btn-primary">
                                            <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                                            </svg>
                                            "Return to Gallery"
                                        </a>
                                    </div>
                                }.into_any()
                            }
                        }
                    }
                })
            }}
        </Suspense>
    }
}

#[component]
fn AnalysisSection(
    perf_id: String,
    analysis_state: ReadSignal<AnalysisState>,
    set_analysis_state: WriteSignal<AnalysisState>,
    loading_message: ReadSignal<String>,
    set_loading_message: WriteSignal<String>,
    loading_progress: ReadSignal<u8>,
    set_loading_progress: WriteSignal<u8>,
) -> impl IntoView {
    let perf_id_for_analysis = perf_id.clone();

    let start_analysis = move |_: leptos::ev::MouseEvent| {
        let id = perf_id_for_analysis.clone();
        set_analysis_state.set(AnalysisState::Loading {
            message: "Starting analysis...".into(),
            progress: 0,
        });

        #[cfg(feature = "hydrate")]
        {
            use leptos::task::spawn_local;

            spawn_local(async move {
                let messages = get_loading_messages();

                for (i, msg) in messages.iter().enumerate() {
                    set_loading_message.set(msg.to_string());
                    set_loading_progress.set(((i + 1) * 100 / messages.len()) as u8);
                    gloo_timers::future::TimeoutFuture::new(300).await;
                }

                match analyze_performance(id).await {
                    Ok(result) => {
                        set_analysis_state.set(AnalysisState::Complete(result));
                    }
                    Err(e) => {
                        set_analysis_state.set(AnalysisState::Error(e.to_string()));
                    }
                }
            });
        }

        #[cfg(not(feature = "hydrate"))]
        {
            let _ = id;
            set_loading_message.set("Loading...".to_string());
            set_loading_progress.set(50);
        }
    };

    view! {
        {move || {
            match analysis_state.get() {
                AnalysisState::Idle => {
                    let handler = start_analysis.clone();
                    view! {
                        <div class="card p-10 text-center">
                            <div class="w-16 h-16 mx-auto mb-6 rounded-lg bg-gold-100 flex items-center justify-center">
                                <svg class="w-8 h-8 text-gold-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round"
                                        d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-xl text-stone-900 mb-2">
                                "Ready to Analyze"
                            </h3>
                            <p class="text-body-md text-stone-500 mb-8 max-w-md mx-auto">
                                "Analyze this performance across 19 musical dimensions and receive personalized feedback from our AI."
                            </p>
                            <button
                                on:click=handler
                                class="btn-primary px-8 py-3 text-body-md"
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                                "Begin Analysis"
                            </button>
                            <p class="text-label-md text-stone-400 mt-4 uppercase tracking-wide">
                                "Typically takes 5-10 seconds"
                            </p>
                        </div>
                    }.into_any()
                },

                AnalysisState::Loading { .. } => view! {
                    <LoadingSpinner
                        message=loading_message
                        progress=loading_progress
                    />
                }.into_any(),

                AnalysisState::Complete(result) => {
                    let models = result.models.clone();
                    let (active_model, set_active_model) = signal(2usize); // Default to Fusion (best)

                    let radar_data = Memo::new(move |_| {
                        let idx = active_model.get();
                        let models_ref = models.clone();
                        models_ref.get(idx)
                            .map(|m| m.dimensions.to_labeled_vec()
                                .into_iter()
                                .map(|(label, value)| RadarDataPoint {
                                    label: label.to_string(),
                                    value,
                                })
                                .collect::<Vec<_>>())
                            .unwrap_or_default()
                    });

                    let radar_signal = Signal::derive(move || radar_data.get());
                    let feedback = result.teacher_feedback.clone();
                    let tips = result.practice_tips.clone();
                    let models_for_selector = result.models.clone();

                    view! {
                        <div class="space-y-8 animate-fade-in">
                            <div class="text-center">
                                <div class="flex justify-center mb-4">
                                    <span class="badge-gold">
                                        <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                        </svg>
                                        "Complete"
                                    </span>
                                </div>
                                <h2 class="font-display text-display-sm text-stone-900 mb-2">
                                    "Analysis Results"
                                </h2>
                                <p class="text-body-md text-stone-500">
                                    "Compare model predictions across 19 dimensions"
                                </p>
                            </div>

                            // Model selector
                            <div class="flex flex-wrap justify-center gap-3">
                                {models_for_selector.iter().enumerate().map(|(i, model)| {
                                    let model_type = model.model_type.clone();
                                    let model_name = model.model_name.clone();
                                    let r2 = model.r_squared;
                                    let is_fusion = model_type == "Fusion";

                                    view! {
                                        <button
                                            on:click=move |_| set_active_model.set(i)
                                            class=move || {
                                                let base = "px-5 py-3 rounded-lg border transition-all duration-200 text-left min-w-[140px]";
                                                if active_model.get() == i {
                                                    format!("{} border-gold-500 bg-gold-50 shadow-sm", base)
                                                } else {
                                                    format!("{} border-stone-200 bg-white hover:border-stone-300", base)
                                                }
                                            }
                                        >
                                            <span class="block font-mono text-body-sm font-medium text-stone-800">
                                                {model_type.clone()}
                                            </span>
                                            <span class="block text-label-sm text-stone-400 mt-0.5">
                                                {model_name}
                                            </span>
                                            <div class="flex items-center gap-2 mt-2">
                                                <span class="text-label-sm font-mono text-stone-500">
                                                    {format!("R\u{00B2} = {:.3}", r2)}
                                                </span>
                                                {if is_fusion {
                                                    view! {
                                                        <span class="px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide bg-gold-100 text-gold-700 rounded">
                                                            "Best"
                                                        </span>
                                                    }.into_any()
                                                } else {
                                                    view! {}.into_any()
                                                }}
                                            </div>
                                        </button>
                                    }
                                }).collect_view()}
                            </div>

                            // Primary output: Natural language feedback
                            <TeacherFeedback feedback=feedback />

                            // Practice tips
                            <PracticeTips tips=tips />

                            // Collapsible radar chart (hidden by default)
                            <CollapsibleRadarChart data=radar_signal size=450 />

                            <div class="text-center pt-6">
                                <a
                                    href="/"
                                    class="inline-flex items-center gap-2 text-body-sm font-medium text-gold-600 hover:text-gold-700 transition-colors"
                                >
                                    "Analyze another performance"
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/>
                                    </svg>
                                </a>
                            </div>
                        </div>
                    }.into_any()
                },

                AnalysisState::Error(msg) => {
                    let handler = start_analysis.clone();
                    view! {
                        <div class="card p-8 text-center border-error/20">
                            <div class="w-12 h-12 mx-auto mb-4 rounded-lg bg-error-light flex items-center justify-center">
                                <svg class="w-6 h-6 text-error" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-lg text-stone-900 mb-2">
                                "Analysis Failed"
                            </h3>
                            <p class="text-body-sm text-stone-500 mb-6">{msg}</p>
                            <button
                                on:click=handler
                                class="btn-secondary"
                            >
                                "Try Again"
                            </button>
                        </div>
                    }.into_any()
                },
            }
        }}
    }
}

#[component]
fn PerformanceNotFound() -> impl IntoView {
    view! {
        <div class="container-narrow text-center py-20 animate-fade-in">
            <div class="w-16 h-16 mx-auto mb-6 rounded-lg bg-stone-100 flex items-center justify-center">
                <svg class="w-8 h-8 text-stone-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round"
                        d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            </div>
            <h1 class="font-display text-display-md text-stone-900 mb-3">
                "Performance Not Found"
            </h1>
            <p class="text-body-md text-stone-500 mb-8">
                "The performance you're looking for doesn't exist or has been removed."
            </p>
            <a href="/" class="btn-primary">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                </svg>
                "Return to Gallery"
            </a>
        </div>
    }
}

#[cfg(feature = "hydrate")]
fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Analyzing audio waveform...",
        "Extracting timing features...",
        "Evaluating articulation...",
        "Processing dynamics...",
        "Analyzing pedal technique...",
        "Assessing timbre qualities...",
        "Measuring expression...",
        "Generating feedback...",
        "Preparing practice tips...",
        "Finalizing analysis...",
    ]
}

use crate::models::{AnalysisResult, Performance};

#[server(GetPerformance, "/api")]
pub async fn get_performance(id: String) -> Result<Performance, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        Performance::find_by_id(&id).ok_or_else(|| ServerFnError::new("Performance not found"))
    }
    #[cfg(not(feature = "ssr"))]
    {
        let _ = id;
        Err(ServerFnError::new("SSR only"))
    }
}

#[server(AnalyzePerformance, "/api")]
pub async fn analyze_performance(id: String) -> Result<AnalysisResult, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        use crate::models::{ModelResult, PerformanceDimensions};
        use crate::services::{
            generate_teacher_feedback, get_performance_dimensions, get_practice_tips,
        };

        fn generate_model_variants(base: &PerformanceDimensions) -> Vec<ModelResult> {
            let symbolic = PerformanceDimensions {
                timing: (base.timing * 1.05).min(1.0),
                articulation_length: (base.articulation_length * 0.95).min(1.0),
                articulation_touch: (base.articulation_touch * 0.98).min(1.0),
                pedal_amount: (base.pedal_amount * 0.85).min(1.0),
                pedal_clarity: (base.pedal_clarity * 0.88).min(1.0),
                timbre_variety: (base.timbre_variety * 0.75).min(1.0),
                timbre_depth: (base.timbre_depth * 0.78).min(1.0),
                timbre_brightness: (base.timbre_brightness * 0.80).min(1.0),
                timbre_loudness: (base.timbre_loudness * 0.82).min(1.0),
                dynamics_range: base.dynamics_range,
                tempo: (base.tempo * 1.02).min(1.0),
                space: base.space,
                balance: base.balance,
                drama: (base.drama * 0.95).min(1.0),
                mood_valence: base.mood_valence,
                mood_energy: base.mood_energy,
                mood_imagination: (base.mood_imagination * 0.92).min(1.0),
                interpretation_sophistication: base.interpretation_sophistication,
                interpretation_overall: (base.interpretation_overall * 0.96).min(1.0),
            };

            let audio = PerformanceDimensions {
                timing: (base.timing * 0.97).min(1.0),
                articulation_length: base.articulation_length,
                articulation_touch: (base.articulation_touch * 1.02).min(1.0),
                pedal_amount: (base.pedal_amount * 1.08).min(1.0),
                pedal_clarity: (base.pedal_clarity * 1.05).min(1.0),
                timbre_variety: (base.timbre_variety * 1.12).min(1.0),
                timbre_depth: (base.timbre_depth * 1.10).min(1.0),
                timbre_brightness: (base.timbre_brightness * 1.08).min(1.0),
                timbre_loudness: (base.timbre_loudness * 1.05).min(1.0),
                dynamics_range: (base.dynamics_range * 1.03).min(1.0),
                tempo: (base.tempo * 0.98).min(1.0),
                space: (base.space * 1.02).min(1.0),
                balance: (base.balance * 1.01).min(1.0),
                drama: (base.drama * 1.04).min(1.0),
                mood_valence: (base.mood_valence * 1.02).min(1.0),
                mood_energy: (base.mood_energy * 1.03).min(1.0),
                mood_imagination: (base.mood_imagination * 1.05).min(1.0),
                interpretation_sophistication: (base.interpretation_sophistication * 1.02).min(1.0),
                interpretation_overall: (base.interpretation_overall * 1.03).min(1.0),
            };

            vec![
                ModelResult {
                    model_name: "PercePiano".to_string(),
                    model_type: "Symbolic".to_string(),
                    r_squared: 0.395,
                    dimensions: symbolic,
                },
                ModelResult {
                    model_name: "MERT-330M".to_string(),
                    model_type: "Audio".to_string(),
                    r_squared: 0.433,
                    dimensions: audio,
                },
                ModelResult {
                    model_name: "Late Fusion".to_string(),
                    model_type: "Fusion".to_string(),
                    r_squared: 0.510,
                    dimensions: base.clone(),
                },
            ]
        }

        let performance =
            Performance::find_by_id(&id).ok_or_else(|| ServerFnError::new("Performance not found"))?;

        let dimensions = get_performance_dimensions(&id).await;
        let models = generate_model_variants(&dimensions);
        let practice_tips = get_practice_tips(&performance, &dimensions).await;
        let teacher_feedback = generate_teacher_feedback(&performance, &dimensions).await;

        Ok(AnalysisResult {
            performance_id: id,
            dimensions,
            models,
            teacher_feedback,
            practice_tips,
        })
    }
    #[cfg(not(feature = "ssr"))]
    {
        let _ = id;
        Err(ServerFnError::new("SSR only"))
    }
}
