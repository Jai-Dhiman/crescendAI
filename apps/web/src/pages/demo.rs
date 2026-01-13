use leptos::prelude::*;
use leptos_router::hooks::use_params_map;

#[cfg(feature = "hydrate")]
use wasm_bindgen::JsCast;

use crate::components::{
    ChatPanel, CollapsibleRadarChart, LoadingSpinner, PracticeTips, RadarDataPoint, TeacherFeedback,
};
use crate::models::{AnalysisResult, AnalysisState, Performance, UploadedPerformance};

#[cfg(feature = "hydrate")]
use crate::components::{AudioPlayer, AudioUpload};

/// Interactive Demo Page - upload your own or use demo recordings
#[component]
pub fn DemoPage() -> impl IntoView {
    let params = use_params_map();
    let initial_id = Memo::new(move |_| params.read().get("id"));

    let (selected_id, set_selected_id) = signal::<Option<String>>(None);
    let (uploaded_perf, set_uploaded_perf) = signal::<Option<UploadedPerformance>>(None);
    let (analysis_state, set_analysis_state) = signal(AnalysisState::Idle);
    let (loading_message, set_loading_message) = signal(String::new());
    let (loading_progress, set_loading_progress) = signal(0u8);

    // Load demo performances for the picker
    let performances_resource = Resource::new(|| (), |_| list_performances());

    // Load selected performance details (for demos only)
    let performance_resource = Resource::new(
        move || {
            // Only fetch if we have a selected_id and it's not an upload
            let id = selected_id.get();
            let is_upload = uploaded_perf.get().map(|u| Some(u.id.clone()) == id).unwrap_or(false);
            if is_upload { None } else { id }
        },
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

    // Handle upload completion
    let on_upload_complete = Callback::new(move |uploaded: UploadedPerformance| {
        let id = uploaded.id.clone();
        set_uploaded_perf.set(Some(uploaded));
        set_selected_id.set(Some(id));
        set_analysis_state.set(AnalysisState::Idle);
    });

    // Select a demo (clears upload)
    let select_demo = move |id: String| {
        set_uploaded_perf.set(None);
        set_selected_id.set(Some(id));
        set_analysis_state.set(AnalysisState::Idle);
    };

    view! {
        <div class="min-h-screen bg-paper-50">
            // Minimal Header
            <header class="border-b border-paper-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
                <div class="container-wide py-4 flex items-center justify-between">
                    <a href="/" class="flex items-center gap-2 text-ink-800 hover:text-sepia-600 transition-colors">
                        <div class="w-8 h-8 rounded-lg bg-sepia-100 flex items-center justify-center">
                            <svg class="w-4 h-4 text-sepia-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                            </svg>
                        </div>
                        <span class="font-display text-heading-md">"CrescendAI"</span>
                    </a>
                    <a
                        href="/"
                        class="text-label-sm text-sepia-600 hover:text-sepia-700 transition-colors"
                    >
                        "Learn More"
                    </a>
                </div>
            </header>

            // Upload Section
            <section class="border-b border-paper-200 bg-white py-6">
                <div class="container-wide">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-label-sm uppercase tracking-wider text-sepia-600">
                            "Upload Your Recording"
                        </h2>
                    </div>

                    <UploadZone on_upload=on_upload_complete />
                </div>
            </section>

            // Demo Selector - Horizontal compact cards
            <section class="border-b border-paper-200 bg-paper-50 py-6">
                <div class="container-wide">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-label-sm uppercase tracking-wider text-sepia-600">
                            "Or Select a Demo Performance"
                        </h2>
                    </div>

                    <Suspense fallback=|| view! {
                        <div class="flex gap-4">
                            {(0..3).map(|_| view! {
                                <div class="flex-1 h-20 rounded-lg bg-paper-100 skeleton"></div>
                            }).collect_view()}
                        </div>
                    }>
                        {move || performances_resource.get().map(|result| {
                            match result {
                                Ok(performances) => {
                                    view! {
                                        <div class="flex flex-col sm:flex-row gap-3">
                                            {performances.into_iter().map(|perf| {
                                                let perf_id = perf.id.clone();
                                                let perf_id_for_click = perf_id.clone();
                                                let is_selected = Signal::derive(move || {
                                                    uploaded_perf.get().is_none() && selected_id.get() == Some(perf_id.clone())
                                                });
                                                view! {
                                                    <CompactRecordingCard
                                                        performance=perf
                                                        is_selected=is_selected
                                                        on_click=move |_| select_demo(perf_id_for_click.clone())
                                                    />
                                                }
                                            }).collect_view()}
                                        </div>
                                    }.into_any()
                                },
                                Err(e) => view! {
                                    <p class="text-body-sm text-error">"Error: " {e.to_string()}</p>
                                }.into_any(),
                            }
                        })}
                    </Suspense>
                </div>
            </section>

            // Main Content Area
            <main class="container-wide py-8">
                <Show
                    when=move || selected_id.get().is_some()
                    fallback=|| view! {
                        <div class="text-center py-20">
                            <div class="w-16 h-16 mx-auto mb-6 rounded-full bg-sepia-50 flex items-center justify-center">
                                <svg class="w-8 h-8 text-sepia-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                                </svg>
                            </div>
                            <h2 class="font-display text-heading-xl text-ink-800 mb-2">
                                "Select a Recording"
                            </h2>
                            <p class="text-body-md text-ink-500">
                                "Choose a performance above to analyze and explore"
                            </p>
                        </div>
                    }
                >
                    {move || {
                        let perf_id = selected_id.get().unwrap_or_default();
                        view! {
                            <AnalysisView
                                perf_id=perf_id
                                uploaded_perf=uploaded_perf
                                performance_resource=performance_resource
                                analysis_state=analysis_state
                                set_analysis_state=set_analysis_state
                                loading_message=loading_message
                                set_loading_message=set_loading_message
                                loading_progress=loading_progress
                                set_loading_progress=set_loading_progress
                            />
                        }
                    }}
                </Show>
            </main>
        </div>
    }
}

/// Compact horizontal recording card
#[component]
fn CompactRecordingCard(
    performance: Performance,
    #[prop(into)]
    is_selected: Signal<bool>,
    on_click: impl Fn(leptos::ev::MouseEvent) + 'static,
) -> impl IntoView {
    let duration = performance.duration_seconds;
    let mins = duration / 60;
    let secs = duration % 60;

    view! {
        <button
            on:click=on_click
            class=move || {
                let base = "flex-1 p-4 rounded-xl border-2 text-left transition-all duration-200 group";
                if is_selected.get() {
                    format!("{} border-sepia-500 bg-sepia-50 shadow-md", base)
                } else {
                    format!("{} border-paper-200 bg-white hover:border-sepia-300 hover:shadow-sm", base)
                }
            }
        >
            <div class="flex items-center gap-4">
                // Play indicator
                <div class=move || {
                    let base = "w-10 h-10 rounded-full flex items-center justify-center transition-colors";
                    if is_selected.get() {
                        format!("{} bg-sepia-600", base)
                    } else {
                        format!("{} bg-paper-200 group-hover:bg-sepia-100", base)
                    }
                }>
                    <svg
                        class=move || {
                            let base = "w-4 h-4 ml-0.5";
                            if is_selected.get() {
                                format!("{} text-white", base)
                            } else {
                                format!("{} text-sepia-600", base)
                            }
                        }
                        fill="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </div>

                // Info
                <div class="flex-1 min-w-0">
                    <p class="text-label-sm text-sepia-600 uppercase tracking-wider truncate">
                        {performance.composer.clone()}
                    </p>
                    <p class="font-display text-heading-sm text-ink-800 truncate">
                        {performance.piece_title.clone()}
                    </p>
                    <p class="text-body-sm text-ink-500 truncate">
                        {performance.performer.clone()}
                    </p>
                </div>

                // Duration
                <span class="text-label-sm text-ink-400 font-mono shrink-0">
                    {format!("{}:{:02}", mins, secs)}
                </span>
            </div>
        </button>
    }
}

/// Upload zone wrapper that handles hydrate/SSR conditional rendering
#[component]
fn UploadZone(
    #[prop(into)]
    on_upload: Callback<UploadedPerformance>,
) -> impl IntoView {
    #[cfg(feature = "hydrate")]
    {
        view! {
            <AudioUpload on_upload=on_upload />
        }
    }

    #[cfg(not(feature = "hydrate"))]
    {
        let _ = on_upload;
        view! {
            <div class="border-2 border-dashed border-paper-300 rounded-lg p-8 text-center">
                <div class="flex flex-col items-center gap-3">
                    <svg class="w-10 h-10 text-paper-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p class="text-body-sm text-ink-400">"Upload component loading..."</p>
                </div>
            </div>
        }
    }
}

#[component]
fn AnalysisView(
    perf_id: String,
    uploaded_perf: ReadSignal<Option<UploadedPerformance>>,
    performance_resource: Resource<Option<Performance>>,
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
                    gloo_timers::future::TimeoutFuture::new(250).await;
                }

                match analyze_performance_via_api(id).await {
                    Ok(result) => {
                        set_analysis_state.set(AnalysisState::Complete(result));
                    }
                    Err(e) => {
                        set_analysis_state.set(AnalysisState::Error(e));
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

    let perf_id_for_view = perf_id.clone();
    let perf_id_for_demo = perf_id;
    let start_analysis_for_upload = start_analysis.clone();
    let start_analysis_for_demo = start_analysis;

    view! {
        {move || {
            let current_perf_id = perf_id_for_view.clone();
            let demo_perf_id = perf_id_for_demo.clone();
            let upload_handler = start_analysis_for_upload.clone();
            let demo_handler = start_analysis_for_demo.clone();

            // If this is an uploaded performance, show it directly
            if let Some(uploaded) = uploaded_perf.get() {
                if uploaded.id == current_perf_id {
                    #[cfg(feature = "hydrate")]
                    let audio_section = {
                        view! {
                            <AudioPlayer
                                audio_url=uploaded.audio_url.clone()
                                title=uploaded.title.clone()
                            />
                        }
                    };

                    #[cfg(not(feature = "hydrate"))]
                    let audio_section = {
                        view! {
                            <div class="card p-6">
                                <h3 class="font-display text-heading-md text-ink-800 mb-4">
                                    {uploaded.title.clone()}
                                </h3>
                                <div class="h-16 bg-paper-100 rounded-lg flex items-center justify-center border border-paper-200">
                                    <p class="text-body-sm text-ink-400">"Audio player loading..."</p>
                                </div>
                            </div>
                        }
                    };

                    return view! {
                        <div class="space-y-6 animate-fade-in">
                            // Audio Player for uploaded file
                            {audio_section}

                            // Analysis Section
                            <AnalysisContent
                                perf_id=current_perf_id
                                analysis_state=analysis_state
                                start_analysis=upload_handler
                                loading_message=loading_message
                                loading_progress=loading_progress
                            />
                        </div>
                    }.into_any();
                }
            }

            // Otherwise show demo performance from resource
            view! {
                <Suspense fallback=|| view! {
                    <div class="space-y-6">
                        <div class="h-20 bg-paper-100 skeleton rounded-xl"></div>
                        <div class="h-64 bg-paper-100 skeleton rounded-xl"></div>
                    </div>
                }>
                    {move || performance_resource.get().map(|perf_opt| {
                        match perf_opt {
                            Some(perf) => {
                                let perf_for_audio = perf.clone();

                                #[cfg(feature = "hydrate")]
                                let audio_section = {
                                    view! {
                                        <AudioPlayer
                                            audio_url=perf_for_audio.audio_url.clone()
                                            title=format!("{} - {}", perf_for_audio.piece_title, perf_for_audio.performer)
                                        />
                                    }
                                };

                                #[cfg(not(feature = "hydrate"))]
                                let audio_section = {
                                    view! {
                                        <div class="card p-6">
                                            <h3 class="font-display text-heading-md text-ink-800 mb-4">
                                                {format!("{} - {}", perf_for_audio.piece_title, perf_for_audio.performer)}
                                            </h3>
                                            <div class="h-16 bg-paper-100 rounded-lg flex items-center justify-center border border-paper-200">
                                                <p class="text-body-sm text-ink-400">"Audio player loading..."</p>
                                            </div>
                                        </div>
                                    }
                                };

                                view! {
                                    <div class="space-y-6 animate-fade-in">
                                        // Audio Player
                                        {audio_section}

                                        // Analysis Section
                                        <AnalysisContent
                                            perf_id=demo_perf_id.clone()
                                            analysis_state=analysis_state
                                            start_analysis=demo_handler.clone()
                                            loading_message=loading_message
                                            loading_progress=loading_progress
                                        />
                                    </div>
                                }.into_any()
                            },
                            None => view! {
                                <div class="text-center py-12">
                                    <p class="text-body-md text-ink-500">"Recording not found"</p>
                                </div>
                            }.into_any(),
                        }
                    })}
                </Suspense>
            }.into_any()
        }}
    }
}

#[component]
fn AnalysisContent(
    perf_id: String,
    analysis_state: ReadSignal<AnalysisState>,
    start_analysis: impl Fn(leptos::ev::MouseEvent) + 'static + Clone + Send,
    loading_message: ReadSignal<String>,
    loading_progress: ReadSignal<u8>,
) -> impl IntoView {
    view! {
        {move || {
            let handler = start_analysis.clone();
            let pid = perf_id.clone();
            match analysis_state.get() {
                AnalysisState::Idle => {
                    view! {
                        <div class="card p-8 text-center">
                            <div class="w-14 h-14 mx-auto mb-5 rounded-xl bg-sepia-100 flex items-center justify-center">
                                <svg class="w-7 h-7 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round"
                                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                            </div>

                            <h3 class="font-display text-heading-lg text-ink-900 mb-2">
                                "Ready to Analyze"
                            </h3>
                            <p class="text-body-md text-ink-500 mb-6 max-w-md mx-auto">
                                "Get AI-powered feedback grounded in piano pedagogy sources"
                            </p>

                            <button
                                on:click=handler
                                class="btn-primary px-6 py-2.5"
                            >
                                "Analyze Performance"
                            </button>
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
                        <AnalysisResults result=result perf_id=pid />
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
fn AnalysisResults(result: AnalysisResult, perf_id: String) -> impl IntoView {
    let models = result.models.clone();
    let (active_model, set_active_model) = signal(2usize); // Default to Fusion (best overall)

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
        <div class="space-y-6 animate-fade-in">
            // Two-column layout: Radar + Model Selector
            <div class="grid lg:grid-cols-5 gap-6">
                // Radar Chart (takes 3 cols)
                <div class="lg:col-span-3 card p-6">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="font-display text-heading-md text-ink-800">
                            "Performance Analysis"
                        </h3>
                        // Model selector pills
                        <div class="flex gap-1 bg-paper-100 p-1 rounded-lg">
                            {models_for_selector.iter().enumerate().map(|(i, model)| {
                                let model_type = model.model_type.clone();
                                view! {
                                    <button
                                        on:click=move |_| set_active_model.set(i)
                                        class=move || {
                                            let base = "px-3 py-1.5 text-label-sm font-medium rounded-md transition-all";
                                            if active_model.get() == i {
                                                format!("{} bg-white text-ink-800 shadow-sm", base)
                                            } else {
                                                format!("{} text-ink-500 hover:text-ink-700", base)
                                            }
                                        }
                                    >
                                        {model_type}
                                    </button>
                                }
                            }).collect_view()}
                        </div>
                    </div>
                    <div class="flex justify-center">
                        <CollapsibleRadarChart data=radar_signal size=380 />
                    </div>
                </div>

                // Practice Tips (takes 2 cols)
                <div class="lg:col-span-2">
                    <PracticeTips tips=tips />
                </div>
            </div>

            // Teacher Feedback with expandable citations
            <TeacherFeedback feedback=feedback />

            // Chat Panel for follow-up questions
            <ChatPanel performance_id=perf_id />
        </div>
    }
}

#[cfg(feature = "hydrate")]
fn get_loading_messages() -> Vec<&'static str> {
    vec![
        "Analyzing audio...",
        "Evaluating timing...",
        "Processing dynamics...",
        "Assessing expression...",
        "Generating feedback...",
        "Preparing results...",
    ]
}

// Server functions
use crate::pages::performance::get_performance;

/// Analyze performance by calling the API endpoint directly.
/// This uses the full analysis pipeline with HF inference and RAG.
#[cfg(feature = "hydrate")]
async fn analyze_performance_via_api(id: String) -> Result<AnalysisResult, String> {
    let window = web_sys::window().ok_or("No window")?;

    let url = format!("/api/analyze/{}", id);
    let resp = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&url))
        .await
        .map_err(|e| format!("Fetch error: {:?}", e))?;

    let response: web_sys::Response = resp.dyn_into().map_err(|_| "Invalid response")?;

    if !response.ok() {
        let status = response.status();
        let text = wasm_bindgen_futures::JsFuture::from(response.text().map_err(|_| "No text")?)
            .await
            .map_err(|_| "Text error")?
            .as_string()
            .unwrap_or_default();

        // Try to parse error message from JSON
        if let Ok(err) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(msg) = err.get("error").and_then(|v| v.as_str()) {
                return Err(msg.to_string());
            }
        }
        return Err(format!("Analysis failed ({})", status));
    }

    let json = wasm_bindgen_futures::JsFuture::from(response.json().map_err(|_| "No JSON")?)
        .await
        .map_err(|_| "JSON error")?;

    let result: AnalysisResult = serde_wasm_bindgen::from_value(json)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    Ok(result)
}

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
