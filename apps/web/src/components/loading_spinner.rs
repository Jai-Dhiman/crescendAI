use leptos::prelude::*;

#[component]
pub fn LoadingSpinner(
    #[prop(into)] message: Signal<String>,
    #[prop(into)] progress: Signal<u8>,
) -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center py-16">
            // Animated spinner
            <div class="relative w-24 h-24 mb-8">
                <div class="absolute inset-0 border-4 border-white/10 rounded-full" />
                <svg class="absolute inset-0 w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                    <circle
                        cx="50"
                        cy="50"
                        r="46"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="8"
                        stroke-linecap="round"
                        class="text-rose-500"
                        stroke-dasharray="289"
                        stroke-dashoffset=move || {
                            let p = progress.get() as f64;
                            289.0 - (289.0 * p / 100.0)
                        }
                        style="transition: stroke-dashoffset 0.3s ease"
                    />
                </svg>
                <div class="absolute inset-0 flex items-center justify-center">
                    <span class="text-xl font-bold text-white">
                        {move || format!("{}%", progress.get())}
                    </span>
                </div>
            </div>

            // Message
            <p class="text-white/70 text-center max-w-md text-lg">
                {move || message.get()}
            </p>

            // Subtle animation hint
            <div class="mt-6 flex gap-1">
                <div class="w-2 h-2 rounded-full bg-rose-500 animate-bounce" style="animation-delay: 0ms" />
                <div class="w-2 h-2 rounded-full bg-rose-500 animate-bounce" style="animation-delay: 150ms" />
                <div class="w-2 h-2 rounded-full bg-rose-500 animate-bounce" style="animation-delay: 300ms" />
            </div>
        </div>
    }
}
