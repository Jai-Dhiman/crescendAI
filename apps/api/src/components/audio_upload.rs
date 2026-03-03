//! Audio upload component with drag-and-drop support.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{File, HtmlInputElement};

use crate::models::UploadedPerformance;

/// Upload state
#[derive(Clone, Debug, PartialEq)]
pub enum UploadState {
    Idle,
    Uploading,
    Complete(UploadedPerformance),
    Error(String),
}

/// Audio upload component with drag-and-drop
#[component]
pub fn AudioUpload(
    /// Callback when upload completes successfully
    #[prop(into)]
    on_upload: Callback<UploadedPerformance>,
) -> impl IntoView {
    let (state, set_state) = signal(UploadState::Idle);
    let (is_dragging, set_is_dragging) = signal(false);
    let file_input_ref = NodeRef::<leptos::html::Input>::new();

    // Handle file selection
    let handle_file = move |file: File| {
        let file_name = file.name();
        let file_size = file.size() as usize;
        let file_type = file.type_();

        // Validate file type
        let allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/webm"];
        if !allowed_types.iter().any(|&t| file_type.contains(t) || file_type.is_empty()) {
            set_state.set(UploadState::Error("Unsupported format. Use MP3, WAV, M4A, or WebM.".to_string()));
            return;
        }

        // Validate file size (50MB)
        if file_size > 50 * 1024 * 1024 {
            set_state.set(UploadState::Error("File too large. Maximum size is 50MB.".to_string()));
            return;
        }

        set_state.set(UploadState::Uploading);

        // Create form data directly with the file
        let form_data = match web_sys::FormData::new() {
            Ok(fd) => fd,
            Err(_) => {
                set_state.set(UploadState::Error("Failed to create form data".to_string()));
                return;
            }
        };

        if let Err(_) = form_data.append_with_blob_and_filename("file", &file, &file_name) {
            set_state.set(UploadState::Error("Failed to append file".to_string()));
            return;
        }

        // Make upload request
        let on_upload = on_upload.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match upload_file(form_data).await {
                Ok(uploaded) => {
                    on_upload.run(uploaded.clone());
                    set_state.set(UploadState::Complete(uploaded));
                }
                Err(e) => {
                    set_state.set(UploadState::Error(e));
                }
            }
        });
    };

    // Handle input change
    let on_input_change = move |ev: leptos::ev::Event| {
        let input: HtmlInputElement = event_target(&ev);
        if let Some(files) = input.files() {
            if let Some(file) = files.get(0) {
                handle_file(file);
            }
        }
    };

    // Handle drag events
    let on_drag_over = move |ev: leptos::ev::DragEvent| {
        ev.prevent_default();
        set_is_dragging.set(true);
    };

    let on_drag_leave = move |_: leptos::ev::DragEvent| {
        set_is_dragging.set(false);
    };

    let on_drop = move |ev: leptos::ev::DragEvent| {
        ev.prevent_default();
        set_is_dragging.set(false);

        if let Some(dt) = ev.data_transfer() {
            if let Some(files) = dt.files() {
                if let Some(file) = files.get(0) {
                    handle_file(file);
                }
            }
        }
    };

    // Click to open file picker
    let on_click = move |_| {
        if let Some(input) = file_input_ref.get() {
            input.click();
        }
    };

    // Reset to idle state
    let reset = move |_| {
        set_state.set(UploadState::Idle);
    };

    view! {
        <div class="w-full">
            // Hidden file input
            <input
                type="file"
                accept="audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/mp4,audio/x-m4a,audio/m4a,audio/webm,.mp3,.wav,.m4a,.webm"
                class="hidden"
                node_ref=file_input_ref
                on:change=on_input_change
            />

            // Upload zone
            {move || match state.get() {
                UploadState::Idle => view! {
                    <div
                        class=move || format!(
                            "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 {} {}",
                            if is_dragging.get() { "border-gold-500 bg-gold-50" } else { "border-stone-300 hover:border-gold-400 hover:bg-stone-50" },
                            "focus-within:ring-2 focus-within:ring-gold-500 focus-within:ring-offset-2"
                        )
                        on:dragover=on_drag_over
                        on:dragleave=on_drag_leave
                        on:drop=on_drop
                        on:click=on_click
                        role="button"
                        tabindex="0"
                        aria-label="Upload audio file"
                    >
                        <div class="flex flex-col items-center gap-3">
                            <svg class="w-10 h-10 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <div class="text-stone-600">
                                <span class="font-medium text-gold-600">"Click to upload"</span>
                                " or drag and drop"
                            </div>
                            <p class="text-sm text-stone-500">
                                "MP3, WAV, M4A, or WebM (max 50MB)"
                            </p>
                        </div>
                    </div>
                }.into_any(),

                UploadState::Uploading => view! {
                    <div class="border-2 border-gold-300 rounded-lg p-8 text-center bg-gold-50">
                        <div class="flex flex-col items-center gap-4">
                            <div class="w-10 h-10 border-4 border-gold-200 border-t-gold-500 rounded-full animate-spin"></div>
                            <p class="text-stone-600 font-medium">"Uploading..."</p>
                        </div>
                    </div>
                }.into_any(),

                UploadState::Complete(ref uploaded) => view! {
                    <div class="border-2 border-green-300 rounded-lg p-6 bg-green-50">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-3">
                                <div class="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                                    <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                                    </svg>
                                </div>
                                <div>
                                    <p class="font-medium text-stone-900">{uploaded.title.clone()}</p>
                                    <p class="text-sm text-stone-500">
                                        {format!("{:.1} MB uploaded", uploaded.file_size_bytes as f64 / 1024.0 / 1024.0)}
                                    </p>
                                </div>
                            </div>
                            <button
                                class="text-stone-400 hover:text-stone-600 p-2"
                                on:click=reset
                                aria-label="Upload a different file"
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    </div>
                }.into_any(),

                UploadState::Error(ref msg) => view! {
                    <div class="border-2 border-red-300 rounded-lg p-6 bg-red-50">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-3">
                                <div class="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                                    <svg class="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                    </svg>
                                </div>
                                <div>
                                    <p class="font-medium text-red-900">"Upload failed"</p>
                                    <p class="text-sm text-red-700">{msg.clone()}</p>
                                </div>
                            </div>
                            <button
                                class="text-red-400 hover:text-red-600 p-2"
                                on:click=reset
                                aria-label="Try again"
                            >
                                "Try again"
                            </button>
                        </div>
                    </div>
                }.into_any(),
            }}
        </div>
    }
}

/// Upload file to /api/upload endpoint
async fn upload_file(form_data: web_sys::FormData) -> Result<UploadedPerformance, String> {
    let window = web_sys::window().ok_or("No window")?;

    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_body(&form_data);

    let request = web_sys::Request::new_with_str_and_init("/api/upload", &opts)
        .map_err(|e| format!("Request error: {:?}", e))?;

    let resp = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
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
        return Err(format!("Upload failed ({})", status));
    }

    let json = wasm_bindgen_futures::JsFuture::from(response.json().map_err(|_| "No JSON")?)
        .await
        .map_err(|_| "JSON error")?;

    let uploaded: UploadedPerformance = serde_wasm_bindgen::from_value(json)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    Ok(uploaded)
}
