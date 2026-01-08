use leptos::prelude::*;

#[component]
pub fn Header() -> impl IntoView {
    view! {
        <header class="sticky top-0 z-50 bg-cream-50/95 backdrop-blur-sm border-b border-stone-200">
            <div class="container-wide">
                <div class="flex items-center justify-between h-16">
                    // Logo and brand
                    <a
                        href="/"
                        class="group flex items-center gap-3 text-stone-800 hover:text-gold-600 transition-colors duration-200"
                        aria-label="CrescendAI Home"
                    >
                        // Logo mark - angular, sophisticated
                        <div class="relative w-10 h-10 bg-gradient-gold rounded-md shadow-gold flex items-center justify-center transition-transform duration-200 group-hover:scale-105">
                            <svg
                                class="w-5 h-5 text-white"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                viewBox="0 0 24 24"
                                aria-hidden="true"
                            >
                                <path d="M9 18V5l12-3v13" />
                                <circle cx="6" cy="18" r="3" />
                                <circle cx="18" cy="15" r="3" />
                            </svg>
                        </div>
                        <div class="hidden sm:block">
                            <span class="font-display text-xl font-semibold tracking-tight text-stone-900">
                                "CrescendAI"
                            </span>
                            <span class="block text-label-sm uppercase tracking-wider text-stone-400">
                                "Performance Analysis"
                            </span>
                        </div>
                    </a>

                    // Navigation
                    <nav class="flex items-center gap-1" role="navigation" aria-label="Main navigation">
                        <a
                            href="/"
                            class="px-4 py-2 text-body-sm font-medium text-stone-600 rounded-md
                                   transition-all duration-200
                                   hover:bg-stone-100 hover:text-stone-800
                                   focus-visible:ring-2 focus-visible:ring-gold-500 focus-visible:ring-offset-2"
                        >
                            "Gallery"
                        </a>
                        <a
                            href="#about"
                            class="px-4 py-2 text-body-sm font-medium text-stone-600 rounded-md
                                   transition-all duration-200
                                   hover:bg-stone-100 hover:text-stone-800
                                   focus-visible:ring-2 focus-visible:ring-gold-500 focus-visible:ring-offset-2"
                        >
                            "About"
                        </a>
                        <div class="w-px h-5 bg-stone-200 mx-2" aria-hidden="true"></div>
                        <a
                            href="#"
                            class="btn-primary text-body-sm"
                        >
                            "Try Demo"
                        </a>
                    </nav>
                </div>
            </div>
        </header>
    }
}
