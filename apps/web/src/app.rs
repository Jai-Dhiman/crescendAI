use leptos::prelude::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;

use crate::components::Header;
use crate::pages::{HomePage, PerformancePage};

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <div class="min-h-screen bg-gradient-page flex flex-col">
                <Header />
                <main class="flex-1 py-8">
                    <Routes fallback=|| view! { <NotFound /> }>
                        <Route path=path!("/") view=HomePage />
                        <Route path=path!("/performance/:id") view=PerformancePage />
                    </Routes>
                </main>
                <Footer />
            </div>
        </Router>
    }
}

#[component]
fn NotFound() -> impl IntoView {
    view! {
        <div class="container-narrow text-center py-20 animate-fade-in">
            <div class="font-display text-[8rem] font-semibold text-stone-100 leading-none mb-4">
                "404"
            </div>
            <h1 class="font-display text-display-md text-stone-900 mb-3">
                "Page Not Found"
            </h1>
            <p class="text-body-md text-stone-500 mb-8">
                "The page you're looking for doesn't exist or has been moved."
            </p>
            <a href="/" class="btn-primary">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                </svg>
                "Return to Gallery"
            </a>
        </div>
    }
}

#[component]
fn Footer() -> impl IntoView {
    view! {
        <footer class="border-t border-stone-200 mt-auto">
            <div class="container-wide py-8">
                <div class="flex flex-col sm:flex-row items-center justify-between gap-4">
                    // Left side - branding
                    <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-md bg-gradient-gold flex items-center justify-center">
                            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M9 18V5l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
                            </svg>
                        </div>
                        <span class="font-display text-lg font-semibold text-stone-700">
                            "CrescendAI"
                        </span>
                    </div>

                    // Center - tech stack
                    <p class="text-body-sm text-stone-500">
                        "Built with Leptos + Rust. Powered by PercePiano."
                    </p>

                    // Right side - copyright
                    <p class="text-label-sm text-stone-400 uppercase tracking-wide">
                        "Research Project"
                    </p>
                </div>
            </div>
        </footer>
    }
}
