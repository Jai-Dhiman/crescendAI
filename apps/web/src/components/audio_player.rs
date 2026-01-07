use leptos::prelude::*;
use leptos::html::Audio;
use wasm_bindgen::prelude::*;

#[component]
pub fn AudioPlayer(
    #[prop(into)] audio_url: String,
    #[prop(into)] title: String,
) -> impl IntoView {
    let audio_ref = NodeRef::<Audio>::new();
    let (is_playing, set_is_playing) = signal(false);
    let (current_time, set_current_time) = signal(0.0f64);
    let (duration, set_duration) = signal(0.0f64);

    // Set up event listeners when audio element is available
    Effect::new(move |_| {
        if let Some(audio) = audio_ref.get() {
            // Loaded metadata - get duration
            let on_loaded = Closure::wrap(Box::new({
                let audio = audio.clone();
                move |_: web_sys::Event| {
                    set_duration.set(audio.duration());
                }
            }) as Box<dyn FnMut(_)>);
            audio.set_onloadedmetadata(Some(on_loaded.as_ref().unchecked_ref()));
            on_loaded.forget();

            // Time update
            let on_timeupdate = Closure::wrap(Box::new({
                let audio = audio.clone();
                move |_: web_sys::Event| {
                    set_current_time.set(audio.current_time());
                }
            }) as Box<dyn FnMut(_)>);
            audio.set_ontimeupdate(Some(on_timeupdate.as_ref().unchecked_ref()));
            on_timeupdate.forget();

            // Ended
            let on_ended = Closure::wrap(Box::new(move |_: web_sys::Event| {
                set_is_playing.set(false);
                set_current_time.set(0.0);
            }) as Box<dyn FnMut(_)>);
            audio.set_onended(Some(on_ended.as_ref().unchecked_ref()));
            on_ended.forget();
        }
    });

    let toggle_play = move |_| {
        if let Some(audio) = audio_ref.get() {
            if is_playing.get() {
                let _ = audio.pause();
                set_is_playing.set(false);
            } else {
                let _ = audio.play();
                set_is_playing.set(true);
            }
        }
    };

    view! {
        <div class="bg-slate-800/50 rounded-xl p-6 border border-white/5">
            <audio
                node_ref=audio_ref
                src=audio_url.clone()
                preload="metadata"
            />

            <h3 class="text-lg font-semibold mb-4 text-white/90">{title}</h3>

            // Waveform placeholder
            <div class="h-16 bg-slate-700/50 rounded-lg mb-4 flex items-center justify-center overflow-hidden">
                <div class="flex items-end gap-0.5 h-12">
                    {(0..60).map(|i| {
                        let height = 20.0 + (i as f64 * 0.17).sin().abs() * 80.0;
                        let delay = i * 20;
                        view! {
                            <div
                                class="w-1 bg-rose-500/60 rounded-full transition-all duration-300"
                                style=format!("height: {}%; animation-delay: {}ms", height, delay)
                            />
                        }
                    }).collect_view()}
                </div>
            </div>

            // Controls
            <div class="flex items-center gap-4">
                <button
                    on:click=toggle_play
                    class="w-12 h-12 rounded-full bg-rose-500 flex items-center justify-center hover:bg-rose-400 transition-colors shadow-lg shadow-rose-500/25"
                >
                    <Show
                        when=move || is_playing.get()
                        fallback=|| view! {
                            <svg class="w-6 h-6 ml-0.5 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M8 5v14l11-7z"/>
                            </svg>
                        }
                    >
                        <svg class="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                        </svg>
                    </Show>
                </button>

                // Progress bar
                <div class="flex-1">
                    <div class="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                            class="h-full bg-rose-500 transition-all duration-100"
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
                    <div class="flex justify-between text-sm text-white/50 mt-1">
                        <span>{move || format_time(current_time.get())}</span>
                        <span>{move || format_time(duration.get())}</span>
                    </div>
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
