use leptos::prelude::*;

#[component]
pub fn LoadingSpinner(
    #[prop(into)] message: Signal<String>,
    #[prop(into)] progress: Signal<u8>,
) -> impl IntoView {
    view! {
        <div
            class="card p-12 flex flex-col items-center justify-center text-center"
            role="status"
            aria-live="polite"
            aria-busy="true"
        >
            <div class="relative w-24 h-24 mb-8">
                <svg class="absolute inset-0 w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                    <circle
                        cx="50"
                        cy="50"
                        r="44"
                        fill="none"
                        stroke="#e7e5e4"
                        stroke-width="6"
                    />
                </svg>
                <svg class="absolute inset-0 w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                    <circle
                        cx="50"
                        cy="50"
                        r="44"
                        fill="none"
                        stroke="url(#loadingGradient)"
                        stroke-width="6"
                        stroke-linecap="round"
                        stroke-dasharray="276.46"
                        stroke-dashoffset=move || {
                            let p = progress.get() as f64;
                            276.46 - (276.46 * p / 100.0)
                        }
                        style="transition: stroke-dashoffset 0.4s cubic-bezier(0.16, 1, 0.3, 1)"
                    />
                    <defs>
                        <linearGradient id="loadingGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#d4a012" />
                            <stop offset="100%" stop-color="#b8860b" />
                        </linearGradient>
                    </defs>
                </svg>
                <div class="absolute inset-0 flex items-center justify-center">
                    <span class="font-display text-display-sm font-semibold text-stone-900">
                        {move || format!("{}", progress.get())}
                    </span>
                    <span class="text-body-sm text-stone-400 ml-0.5">"%"</span>
                </div>
            </div>

            <p class="text-body-md text-stone-600 max-w-sm mb-4">
                {move || message.get()}
            </p>

            <div class="flex gap-1.5" aria-hidden="true">
                <div class="w-2 h-2 rounded-full bg-gold-400 animate-bounce" style="animation-delay: 0ms"></div>
                <div class="w-2 h-2 rounded-full bg-gold-500 animate-bounce" style="animation-delay: 150ms"></div>
                <div class="w-2 h-2 rounded-full bg-gold-600 animate-bounce" style="animation-delay: 300ms"></div>
            </div>

            <span class="sr-only">
                {move || format!("Loading: {}% complete. {}", progress.get(), message.get())}
            </span>
        </div>
    }
}
