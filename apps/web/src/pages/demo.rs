use leptos::prelude::*;
use leptos_router::hooks::use_params_map;

use crate::components::{CollapsibleRadarChart, LoadingSpinner, PracticeTips, RadarDataPoint, TeacherFeedback};
use crate::models::{AnalysisResult, AnalysisState, Performance};

#[cfg(feature = "hydrate")]
use crate::components::AudioPlayer;

/// Demo page for the CrescendAI research showcase
/// Recording picker first, then model comparison with analysis
#[component]
pub fn DemoPage() -> impl IntoView {
    let params = use_params_map();
    let initial_id = Memo::new(move |_| params.read().get("id"));

    let (selected_id, set_selected_id) = signal::<Option<String>>(None);
    let (analysis_state, set_analysis_state) = signal(AnalysisState::Idle);
    let (loading_message, set_loading_message) = signal(String::new());
    let (loading_progress, set_loading_progress) = signal(0u8);

    // Load all performances for the picker
    let performances_resource = Resource::new(|| (), |_| list_performances());

    // Load selected performance details
    let performance_resource = Resource::new(
        move || selected_id.get(),
        |id| async move {
            match id {
                Some(id) => get_performance(id).await.ok(),
                None => None,
            }
        },
    );

    // Set initial ID from URL params
    Effect::new(move || {
        if let Some(id) = initial_id.get() {
            set_selected_id.set(Some(id));
        }
    });

    view! {
        <div class="animate-fade-in">
            // Page Header
            <section class="section-sm border-b border-paper-300 bg-paper-100">
                <div class="container-wide">
                    <nav class="mb-6">
                        <a
                            href="/"
                            class="group inline-flex items-center gap-2 text-body-sm text-sepia-600 hover:text-sepia-700 transition-colors"
                        >
                            <svg
                                class="w-4 h-4 transition-transform group-hover:-translate-x-0.5"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                viewBox="0 0 24 24"
                            >
                                <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                            </svg>
                            "Back to Research Overview"
                        </a>
                    </nav>

                    <h1 class="font-display text-display-lg text-ink-900 mb-2">
                        "Interactive Demo"
                    </h1>
                    <p class="text-body-md text-ink-500 max-w-2xl">
                        "Select a recording to compare model predictions across 19 perceptual dimensions."
                    </p>
                </div>
            </section>

            // Main Content
            <Show
                when=move || selected_id.get().is_none()
                fallback=move || {
                    let perf_id = selected_id.get().unwrap_or_default();
                    view! {
                        <AnalysisView
                            perf_id=perf_id
                            performance_resource=performance_resource
                            analysis_state=analysis_state
                            set_analysis_state=set_analysis_state
                            loading_message=loading_message
                            set_loading_message=set_loading_message
                            loading_progress=loading_progress
                            set_loading_progress=set_loading_progress
                            set_selected_id=set_selected_id
                        />
                    }
                }
            >
                <RecordingPicker
                    performances_resource=performances_resource
                    set_selected_id=set_selected_id
                />
            </Show>
        </div>
    }
}

#[component]
fn RecordingPicker(
    performances_resource: Resource<Result<Vec<Performance>, ServerFnError>>,
    set_selected_id: WriteSignal<Option<String>>,
) -> impl IntoView {
    view! {
        <section class="section">
            <div class="container-wide">
                <div class="text-center mb-12">
                    <div class="step-indicator mb-4">
                        <span class="step-number">"1"</span>
                        "Select a Recording"
                    </div>

                    <h2 class="font-display text-display-sm text-ink-900 mb-4">
                        "Choose a Performance to Analyze"
                    </h2>
                    <p class="text-body-md text-ink-500 max-w-xl mx-auto">
                        "Each recording features a legendary pianist performing a masterwork. Click to see how our models evaluate the performance."
                    </p>
                </div>

                <Suspense fallback=|| view! {
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {(0..6).map(|_| view! {
                            <div class="recording-card">
                                <div class="aspect-[4/3] bg-paper-200 skeleton"></div>
                                <div class="p-5">
                                    <div class="h-4 w-20 bg-paper-200 skeleton mb-3"></div>
                                    <div class="h-6 w-48 bg-paper-200 skeleton mb-2"></div>
                                    <div class="h-4 w-32 bg-paper-200 skeleton"></div>
                                </div>
                            </div>
                        }).collect_view()}
                    </div>
                }>
                    {move || performances_resource.get().map(|result| {
                        match result {
                            Ok(performances) => {
                                view! {
                                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 stagger">
                                        {performances.into_iter().map(|perf| {
                                            let perf_id = perf.id.clone();
                                            view! {
                                                <RecordingCard
                                                    performance=perf
                                                    on_click=move |_| set_selected_id.set(Some(perf_id.clone()))
                                                />
                                            }
                                        }).collect_view()}
                                    </div>
                                }.into_any()
                            },
                            Err(e) => view! {
                                <div class="text-center py-12">
                                    <p class="text-body-md text-error">"Error loading recordings: " {e.to_string()}</p>
                                </div>
                            }.into_any(),
                        }
                    })}
                </Suspense>
            </div>
        </section>
    }
}

#[component]
fn RecordingCard(
    performance: Performance,
    on_click: impl Fn(leptos::ev::MouseEvent) + 'static,
) -> impl IntoView {
    let duration = performance.duration_seconds;
    let mins = duration / 60;
    let secs = duration % 60;

    view! {
        <button
            on:click=on_click
            class="recording-card group text-left w-full focus:outline-none"
        >
            // Thumbnail area with play icon
            <div class="aspect-[4/3] bg-gradient-sepia-subtle flex items-center justify-center relative overflow-hidden">
                // Piano keys pattern (subtle decorative element)
                <div class="absolute inset-x-0 bottom-0 h-8 flex">
                    {(0..12).map(|i| {
                        let is_black = matches!(i, 1 | 3 | 6 | 8 | 10);
                        if is_black {
                            view! {
                                <div class="w-[6%] h-5 bg-ink-800 rounded-b-sm -ml-[3%] z-10 opacity-20"></div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="flex-1 bg-paper-50 border-r border-paper-300 opacity-30"></div>
                            }.into_any()
                        }
                    }).collect_view()}
                </div>

                // Play icon
                <div class="w-16 h-16 rounded-full bg-paper-50/90 flex items-center justify-center
                            shadow-lg transition-all duration-300
                            group-hover:bg-sepia-600 group-hover:scale-110">
                    <svg
                        class="w-6 h-6 text-sepia-600 ml-1 transition-colors group-hover:text-paper-50"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </div>

                // Duration badge
                <div class="absolute bottom-2 right-2 px-2 py-1 bg-ink-800/80 text-paper-50 text-label-sm rounded">
                    {format!("{}:{:02}", mins, secs)}
                </div>
            </div>

            // Info section
            <div class="p-5">
                <span class="text-label-sm uppercase tracking-wider text-sepia-600 mb-1 block">
                    {performance.composer.clone()}
                </span>

                <h3 class="font-display text-heading-lg text-ink-800 mb-1 line-clamp-2 group-hover:text-sepia-700 transition-colors">
                    {performance.piece_title.clone()}
                </h3>

                <p class="text-body-sm text-sepia-600 font-medium">
                    {performance.performer.clone()}
                </p>

                {performance.year_recorded.map(|y| view! {
                    <p class="text-label-sm text-ink-400 mt-2">{format!("Recorded {}", y)}</p>
                })}
            </div>
        </button>
    }
}

#[component]
fn AnalysisView(
    perf_id: String,
    performance_resource: Resource<Option<Performance>>,
    analysis_state: ReadSignal<AnalysisState>,
    set_analysis_state: WriteSignal<AnalysisState>,
    loading_message: ReadSignal<String>,
    set_loading_message: WriteSignal<String>,
    loading_progress: ReadSignal<u8>,
    set_loading_progress: WriteSignal<u8>,
    set_selected_id: WriteSignal<Option<String>>,
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

    let on_change = move || {
        set_selected_id.set(None);
        set_analysis_state.set(AnalysisState::Idle);
    };

    view! {
        <Suspense fallback=|| view! {
            <div class="container-wide py-10">
                <div class="h-8 w-32 bg-paper-200 skeleton mb-4"></div>
                <div class="h-12 w-96 bg-paper-200 skeleton mb-8"></div>
                <div class="card h-48 bg-paper-100 skeleton"></div>
            </div>
        }>
            {move || performance_resource.get().map(|perf_opt| {
                match perf_opt {
                    Some(perf) => {
                        let perf_for_header = perf.clone();

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
                                        <h3 class="font-display text-heading-md text-ink-800 mb-5">
                                            {format!("{} - {}", perf_for_audio.piece_title, perf_for_audio.performer)}
                                        </h3>
                                        <div class="h-20 bg-paper-100 rounded-lg mb-5 flex items-center justify-center border border-paper-200">
                                            <p class="text-body-sm text-ink-400">"Audio player loading..."</p>
                                        </div>
                                    </div>
                                </div>
                            }
                        };

                        view! {
                            <div class="container-wide py-10 animate-fade-in">
                                // Performance Header with change button
                                <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8 pb-8 border-b border-paper-300">
                                    <div>
                                        <div class="flex items-center gap-3 mb-3">
                                            <span class="badge-sepia">{perf_for_header.composer.clone()}</span>
                                            {perf_for_header.year_recorded.map(|year| view! {
                                                <span class="badge-neutral">{format!("{}", year)}</span>
                                            })}
                                        </div>
                                        <h2 class="font-display text-display-sm md:text-display-md text-ink-900 mb-2">
                                            {perf_for_header.piece_title.clone()}
                                        </h2>
                                        <p class="text-heading-md text-sepia-600">
                                            {perf_for_header.performer.clone()}
                                        </p>
                                    </div>
                                    <button
                                        on:click=move |_| on_change()
                                        class="btn-secondary text-body-sm shrink-0"
                                    >
                                        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/>
                                        </svg>
                                        "Change Recording"
                                    </button>
                                </div>

                                // Audio Player
                                {audio_section}

                                // Analysis Section
                                <AnalysisContent
                                    _perf_id=perf_id.clone()
                                    analysis_state=analysis_state
                                    start_analysis=start_analysis.clone()
                                    loading_message=loading_message
                                    loading_progress=loading_progress
                                />
                            </div>
                        }.into_any()
                    },
                    None => view! {
                        <div class="container-narrow text-center py-20">
                            <p class="text-body-md text-ink-500">"Recording not found"</p>
                            <button
                                on:click=move |_| on_change()
                                class="btn-primary mt-6"
                            >
                                "Choose Another Recording"
                            </button>
                        </div>
                    }.into_any(),
                }
            })}
        </Suspense>
    }
}

#[component]
fn AnalysisContent(
    _perf_id: String,
    analysis_state: ReadSignal<AnalysisState>,
    start_analysis: impl Fn(leptos::ev::MouseEvent) + 'static + Clone + Send,
    loading_message: ReadSignal<String>,
    loading_progress: ReadSignal<u8>,
) -> impl IntoView {
    view! {
        {move || {
            let handler = start_analysis.clone();
            match analysis_state.get() {
                AnalysisState::Idle => {
                    view! {
                        <div class="card p-10 text-center">
                            <div class="step-indicator mb-6 mx-auto w-fit">
                                <span class="step-number">"2"</span>
                                "Analyze Performance"
                            </div>

                            <div class="w-16 h-16 mx-auto mb-6 rounded-xl bg-sepia-100 flex items-center justify-center">
                                <svg class="w-8 h-8 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round"
                                        d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                </svg>
                            </div>

                            <h3 class="font-display text-heading-xl text-ink-900 mb-3">
                                "Ready to Analyze"
                            </h3>
                            <p class="text-body-md text-ink-500 mb-8 max-w-md mx-auto">
                                "Compare predictions from Symbolic, Audio, and Fusion models across 19 perceptual dimensions."
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

                            <p class="text-label-md text-ink-400 mt-4 uppercase tracking-wider">
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
                    view! {
                        <AnalysisResults result=result />
                    }.into_any()
                },

                AnalysisState::Error(msg) => {
                    view! {
                        <div class="card p-8 text-center border-error/20">
                            <div class="w-12 h-12 mx-auto mb-4 rounded-lg bg-error-light flex items-center justify-center">
                                <svg class="w-6 h-6 text-error" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <h3 class="font-display text-heading-lg text-ink-900 mb-2">
                                "Analysis Failed"
                            </h3>
                            <p class="text-body-sm text-ink-500 mb-6">{msg}</p>
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
fn AnalysisResults(result: AnalysisResult) -> impl IntoView {
    let models = result.models.clone();
    let (active_model, set_active_model) = signal(1usize); // Default to Audio (best individual)

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
        <div class="space-y-10 animate-fade-in">
            // Success banner
            <div class="text-center">
                <div class="flex justify-center mb-4">
                    <span class="badge-sepia">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                        </svg>
                        "Analysis Complete"
                    </span>
                </div>
            </div>

            // Step 2: Model Comparison
            <div>
                <div class="step-indicator mb-6">
                    <span class="step-number">"2"</span>
                    "Compare Models"
                </div>

                <h3 class="font-display text-heading-xl text-ink-900 mb-6">
                    "Model Predictions"
                </h3>

                <div class="flex flex-wrap gap-4 mb-8">
                    {models_for_selector.iter().enumerate().map(|(i, model)| {
                        let model_type = model.model_type.clone();
                        let model_name = model.model_name.clone();
                        let r2 = model.r_squared;
                        let is_audio = model_type == "Audio";

                        view! {
                            <button
                                on:click=move |_| set_active_model.set(i)
                                class=move || {
                                    let base = "px-6 py-4 rounded-lg border transition-all duration-300 text-left min-w-[160px]";
                                    if active_model.get() == i {
                                        if is_audio {
                                            format!("{} border-sepia-500 bg-sepia-50 shadow-sepia", base)
                                        } else {
                                            format!("{} border-ink-300 bg-paper-100 shadow-sm", base)
                                        }
                                    } else {
                                        format!("{} border-paper-300 bg-paper-50 hover:border-sepia-300 hover:shadow-sm", base)
                                    }
                                }
                            >
                                <span class="block font-mono text-body-sm font-medium text-ink-800">
                                    {model_type.clone()}
                                </span>
                                <span class="block text-label-sm text-ink-400 mt-0.5">
                                    {model_name}
                                </span>
                                <div class="flex items-center gap-2 mt-3">
                                    <span class="font-mono text-label-md text-ink-500">
                                        {format!("R\u{00B2} = {:.3}", r2)}
                                    </span>
                                    {if is_audio {
                                        view! {
                                            <span class="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-sepia-200 text-sepia-700 rounded">
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

                // Radar Chart
                <div class="card p-8">
                    <h4 class="font-display text-heading-lg text-ink-800 mb-6 text-center">
                        "19-Dimension Analysis"
                    </h4>
                    <div class="flex justify-center">
                        <CollapsibleRadarChart data=radar_signal size=450 />
                    </div>
                </div>
            </div>

            // Step 3: Applications
            <div class="border-t border-paper-300 pt-10">
                <div class="step-indicator mb-6">
                    <span class="step-number">"3"</span>
                    "Applications"
                </div>

                <h3 class="font-display text-heading-xl text-ink-900 mb-3">
                    "Downstream Applications"
                </h3>
                <p class="text-body-md text-ink-500 mb-8 max-w-2xl">
                    "The 19-dimension analysis enables practical applications for music education. Below are example outputs demonstrating how the model predictions can be applied."
                </p>

                <div class="grid md:grid-cols-2 gap-8">
                    <TeacherFeedback feedback=feedback />
                    <PracticeTips tips=tips />
                </div>
            </div>

            // Footer link
            <div class="text-center pt-6 border-t border-paper-300">
                <a
                    href="/"
                    class="inline-flex items-center gap-2 text-body-sm font-medium text-sepia-600 hover:text-sepia-700 transition-colors"
                >
                    "Back to Research Overview"
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/>
                    </svg>
                </a>
            </div>
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

// Server functions
use crate::pages::performance::get_performance;
#[cfg(feature = "hydrate")]
use crate::pages::performance::analyze_performance;

#[server(ListPerformances, "/api")]
pub async fn list_performances() -> Result<Vec<Performance>, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        Ok(Performance::get_demo_performances())
    }
    #[cfg(not(feature = "ssr"))]
    {
        Err(ServerFnError::new("SSR only"))
    }
}
