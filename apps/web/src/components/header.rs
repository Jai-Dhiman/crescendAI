use leptos::prelude::*;

#[component]
pub fn Header() -> impl IntoView {
    view! {
        <header class="py-10 lg:py-16">
            <div class="container-editorial">
                // Three-column masthead
                <div class="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 lg:gap-8 items-start text-center lg:text-left">
                    // Left: navigation links
                    <nav class="flex flex-row lg:flex-col items-center lg:items-start justify-center lg:justify-start gap-4 lg:gap-1" role="navigation" aria-label="Main navigation">
                        <a href="/#how-it-works" class="editorial-nav-link">
                            "How It Works"
                        </a>
                        <a href="/analyze" class="editorial-nav-link">
                            "Analyze"
                        </a>
                        <a
                            href="https://arxiv.org/abs/2601.19029"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="inline-flex items-center gap-1.5 editorial-nav-link"
                        >
                            "Paper"
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                            </svg>
                        </a>
                    </nav>

                    // Center: large serif brand name
                    <a href="/" class="font-display text-display-xl lg:text-display-3xl text-ink-900 tracking-tight">
                        "Crescend"
                    </a>

                    // Right: tagline
                    <p class="text-body-sm text-ink-500 lg:text-right max-w-xs mx-auto lg:mx-0 lg:ml-auto">
                        "Record yourself playing. Get the feedback a great teacher would give you."
                    </p>
                </div>
            </div>

            // Thin rule below masthead
            <div class="container-editorial mt-10 lg:mt-16">
                <hr class="editorial-rule" />
            </div>
        </header>
    }
}
