use leptos::prelude::*;

/// Text input for asking follow-up questions
#[component]
pub fn ChatInput(
    /// Callback when user submits a question
    on_submit: impl Fn(String) + Send + Sync + 'static + Clone,
    /// Whether the input is disabled (e.g., while loading)
    #[prop(default = false)]
    disabled: bool,
    /// Placeholder text
    #[prop(default = "Ask about your performance...")]
    placeholder: &'static str,
) -> impl IntoView {
    let (input_value, set_input_value) = signal(String::new());
    let on_submit_clone = on_submit.clone();

    let handle_submit = move || {
        let value = input_value.get();
        if !value.trim().is_empty() {
            on_submit_clone(value.trim().to_string());
            set_input_value.set(String::new());
        }
    };

    let handle_submit_click = handle_submit.clone();
    let handle_submit_keydown = handle_submit.clone();

    view! {
        <div class="flex items-center gap-2 p-3 bg-paper-50 border-t border-paper-200">
            <input
                type="text"
                class="flex-1 px-4 py-2.5 text-body-md text-ink-700 bg-white border border-paper-300 rounded-lg placeholder-ink-400 focus:outline-none focus:ring-2 focus:ring-clay-200 focus:border-clay-400 transition-colors disabled:bg-paper-100 disabled:cursor-not-allowed"
                placeholder=placeholder
                prop:value=move || input_value.get()
                on:input=move |ev| {
                    set_input_value.set(event_target_value(&ev));
                }
                on:keydown=move |ev| {
                    if ev.key() == "Enter" && !ev.shift_key() {
                        ev.prevent_default();
                        handle_submit_keydown();
                    }
                }
                disabled=disabled
            />
            <button
                type="button"
                class="flex items-center justify-center w-10 h-10 bg-clay-600 text-white rounded-lg hover:bg-clay-700 focus:outline-none focus:ring-2 focus:ring-clay-200 transition-colors disabled:bg-clay-300 disabled:cursor-not-allowed"
                on:click=move |_| handle_submit_click()
                disabled=disabled
                aria-label="Send message"
            >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                </svg>
            </button>
        </div>
    }
}
