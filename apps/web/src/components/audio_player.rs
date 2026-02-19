use leptos::html::Audio;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[component]
pub fn AudioPlayer(
    #[prop(into)] audio_url: String,
    #[prop(into)] title: String,
) -> impl IntoView {
    let audio_ref = NodeRef::<Audio>::new();
    let (is_playing, set_is_playing) = signal(false);
    let (current_time, set_current_time) = signal(0.0f64);
    let (duration, set_duration) = signal(0.0f64);

    Effect::new(move |_| {
        if let Some(audio) = audio_ref.get() {
            let on_loaded = Closure::wrap(Box::new({
                let audio = audio.clone();
                move |_: web_sys::Event| {
                    set_duration.set(audio.duration());
                }
            }) as Box<dyn FnMut(_)>);
            audio.set_onloadedmetadata(Some(on_loaded.as_ref().unchecked_ref()));
            on_loaded.forget();

            let on_timeupdate = Closure::wrap(Box::new({
                let audio = audio.clone();
                move |_: web_sys::Event| {
                    set_current_time.set(audio.current_time());
                }
            }) as Box<dyn FnMut(_)>);
            audio.set_ontimeupdate(Some(on_timeupdate.as_ref().unchecked_ref()));
            on_timeupdate.forget();

            let on_ended = Closure::wrap(Box::new(move |_: web_sys::Event| {
                set_is_playing.set(false);
                set_current_time.set(0.0);
            }) as Box<dyn FnMut(_)>);
            audio.set_onended(Some(on_ended.as_ref().unchecked_ref()));
            on_ended.forget();
        }
    });

    let toggle_play = move |_| {
        web_sys::console::log_1(&"Toggle play clicked!".into());
        if let Some(audio) = audio_ref.get() {
            web_sys::console::log_1(&format!("Audio src: {}", audio.src()).into());
            if is_playing.get() {
                let _ = audio.pause();
                set_is_playing.set(false);
            } else {
                // play() returns a Promise that must be awaited
                match audio.play() {
                    Ok(promise) => {
                        spawn_local(async move {
                            match JsFuture::from(promise).await {
                                Ok(_) => {}
                                Err(e) => {
                                    web_sys::console::error_1(&format!("Play promise rejected: {:?}", e).into());
                                }
                            }
                        });
                        set_is_playing.set(true);
                    }
                    Err(e) => {
                        web_sys::console::error_1(&format!("Play error: {:?}", e).into());
                    }
                }
            }
        } else {
            web_sys::console::error_1(&"Audio element not found".into());
        }
    };

    view! {
        <div class="card p-6">
            <audio
                node_ref=audio_ref
                src=audio_url.clone()
                preload="metadata"
                class="hidden"
            />

            <h3 class="font-display text-heading-md text-ink-800 mb-5">{title}</h3>

            <div
                class="h-20 bg-paper-100 rounded-lg mb-5 flex items-center justify-center overflow-hidden border border-paper-200"
                role="img"
                aria-label="Audio waveform visualization"
            >
                <div class="flex items-end gap-0.5 h-14 px-4">
                    {(0..80).map(|i| {
                        let height = 15.0 + (i as f64 * 0.15).sin().abs() * 85.0;
                        view! {
                            <div
                                class="w-1 rounded-full transition-all duration-200"
                                class:bg-clay-500={move || {
                                    let d = duration.get();
                                    if d > 0.0 {
                                        let progress = current_time.get() / d;
                                        (i as f64 / 80.0) < progress
                                    } else {
                                        false
                                    }
                                }}
                                class:bg-paper-300={move || {
                                    let d = duration.get();
                                    if d > 0.0 {
                                        let progress = current_time.get() / d;
                                        (i as f64 / 80.0) >= progress
                                    } else {
                                        true
                                    }
                                }}
                                style=format!("height: {}%", height)
                            />
                        }
                    }).collect_view()}
                </div>
            </div>

            <div class="flex items-center gap-5">
                <button
                    on:click=toggle_play
                    class="w-12 h-12 rounded-full bg-clay-600 flex items-center justify-center
                           shadow-md transition-all duration-200
                           hover:scale-105 hover:bg-clay-700 hover:shadow-lg
                           active:scale-95
                           focus-visible:ring-2 focus-visible:ring-clay-500 focus-visible:ring-offset-2"
                    aria-label={move || if is_playing.get() { "Pause" } else { "Play" }}
                >
                    <Show
                        when=move || is_playing.get()
                        fallback=|| view! {
                            <svg class="w-5 h-5 ml-0.5 text-white" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M8 5v14l11-7z"/>
                            </svg>
                        }
                    >
                        <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                        </svg>
                    </Show>
                </button>

                <div class="flex-1">
                    <div
                        class="h-1.5 bg-paper-200 rounded-full overflow-hidden cursor-pointer"
                        role="progressbar"
                        aria-valuenow={move || current_time.get() as u32}
                        aria-valuemin="0"
                        aria-valuemax={move || duration.get() as u32}
                    >
                        <div
                            class="h-full bg-clay-500 rounded-full transition-all duration-100"
                            style=move || {
                                let d = duration.get();
                                let progress = if d > 0.0 {
                                    (current_time.get() / d) * 100.0
                                } else {
                                    0.0
                                };
                                format!("width: {}%", progress)
                            }
                        />
                    </div>
                    <div class="flex justify-between text-label-sm text-ink-400 mt-2">
                        <span>{move || format_time(current_time.get())}</span>
                        <span>{move || format_time(duration.get())}</span>
                    </div>
                </div>

                <div class="hidden sm:flex items-center gap-1.5 text-ink-400" aria-hidden="true">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" />
                    </svg>
                </div>
            </div>
        </div>
    }
}

fn format_time(seconds: f64) -> String {
    if seconds.is_nan() || seconds.is_infinite() {
        return "0:00".to_string();
    }
    let mins = (seconds / 60.0).floor() as u32;
    let secs = (seconds % 60.0).floor() as u32;
    format!("{}:{:02}", mins, secs)
}
