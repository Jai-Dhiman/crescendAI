use leptos::prelude::*;

#[component]
pub fn Header() -> impl IntoView {
    view! {
        <header class="border-b border-white/10 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
            <div class="container mx-auto px-4 py-4 flex items-center justify-between">
                <a href="/" class="flex items-center gap-3 hover:opacity-80 transition-opacity">
                    <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-500 to-rose-600 flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                    </div>
                    <div>
                        <h1 class="font-serif text-xl font-semibold text-white">"Piano Feedback"</h1>
                        <p class="text-xs text-white/50">"AI-Powered Performance Analysis"</p>
                    </div>
                </a>
                <nav class="flex items-center gap-6">
                    <a href="/" class="text-sm text-white/70 hover:text-white transition-colors">"Gallery"</a>
                    <a href="#about" class="text-sm text-white/70 hover:text-white transition-colors">"About"</a>
                </nav>
            </div>
        </header>
    }
}
