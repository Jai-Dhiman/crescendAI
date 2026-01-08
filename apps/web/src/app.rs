use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;

use crate::components::Header;
use crate::pages::{HomePage, PerformancePage};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/output.css"/>

        <Link rel="preconnect" href="https://fonts.googleapis.com"/>
        <Link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous"/>
        <Link
            href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap"
            rel="stylesheet"
        />

        <Title text="CrescendAI - Piano Performance Analysis"/>
        <Meta name="description" content="AI-powered piano performance analysis across 19 musical dimensions"/>

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
                    <div class="flex items-center gap-3">
                        <img
                            src="/crescendai.png"
                            alt="CrescendAI Logo"
                            class="w-8 h-8 rounded-md"
                        />
                        <span class="font-display text-lg font-semibold text-stone-700">
                            "CrescendAI"
                        </span>
                    </div>

                    <p class="text-body-sm text-stone-500">
                        "Powered by PercePiano"
                    </p>

                    <p class="text-label-sm text-stone-400 uppercase tracking-wide">
                        "Research Project"
                    </p>
                </div>
            </div>
        </footer>
    }
}
