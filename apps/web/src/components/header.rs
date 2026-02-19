use leptos::prelude::*;

#[component]
pub fn Header() -> impl IntoView {
    view! {
        <header class="sticky top-0 z-50 bg-paper-50/95 backdrop-blur-sm border-b border-paper-300">
            <div class="container-wide">
                <div class="flex items-center justify-between h-16">
                    // Logo and brand
                    <a
                        href="/"
                        class="group flex items-center gap-3 text-ink-800 hover:text-clay-600 transition-colors duration-200"
                        aria-label="Crescend Home"
                    >
                        <div class="relative w-10 h-10 rounded-md overflow-hidden transition-transform duration-200 group-hover:scale-105">
                            <img
                                src="/crescendai.png"
                                alt=""
                                class="w-full h-full object-cover"
                                aria-hidden="true"
                            />
                        </div>
                        <span class="hidden sm:block font-display text-xl font-medium tracking-tight text-ink-900">
                            "Crescend"
                        </span>
                    </a>

                    // Navigation
                    <nav class="flex items-center gap-1" role="navigation" aria-label="Main navigation">
                        <a
                            href="/#how-it-works"
                            class="nav-link hidden sm:inline-flex"
                        >
                            "How It Works"
                        </a>
                        <a
                            href="/analyze"
                            class="nav-link"
                        >
                            "Analyze"
                        </a>

                        <div class="w-px h-5 bg-paper-300 mx-2" aria-hidden="true"></div>

                        // Paper link (external)
                        <a
                            href="https://arxiv.org/abs/2601.19029"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="inline-flex items-center gap-1.5 px-4 py-2 text-body-sm font-medium text-ink-600 rounded-md
                                   transition-all duration-200
                                   hover:bg-paper-200 hover:text-ink-800
                                   focus-visible:ring-2 focus-visible:ring-clay-500 focus-visible:ring-offset-2"
                            title="Read the research paper on arXiv"
                        >
                            "Paper"
                            <svg class="w-3.5 h-3.5 text-ink-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                            </svg>
                        </a>
                    </nav>
                </div>
            </div>
        </header>
    }
}
