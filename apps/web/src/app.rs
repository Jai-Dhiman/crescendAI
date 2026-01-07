use leptos::prelude::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;

use crate::components::Header;
use crate::pages::{HomePage, PerformancePage};

#[component]
pub fn App() -> impl IntoView {
    view! {
        <Router>
            <div class="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950">
                <Header />
                <main class="container mx-auto px-4 py-8">
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
        <div class="text-center py-20">
            <h1 class="text-6xl font-bold text-white/20 mb-4">"404"</h1>
            <h2 class="text-2xl font-serif text-white mb-4">"Page Not Found"</h2>
            <p class="text-white/50 mb-6">"The page you're looking for doesn't exist."</p>
            <a
                href="/"
                class="inline-flex items-center gap-2 px-6 py-3 bg-rose-500 hover:bg-rose-400 rounded-lg font-semibold transition-colors"
            >
                "Return to Gallery"
            </a>
        </div>
    }
}

#[component]
fn Footer() -> impl IntoView {
    view! {
        <footer class="border-t border-white/10 mt-20">
            <div class="container mx-auto px-4 py-8 text-center">
                <p class="text-white/40 text-sm">
                    "Built with Leptos + Rust. Powered by the PercePiano model."
                </p>
                <p class="text-white/30 text-xs mt-2">
                    "CrescendAI Research Project"
                </p>
            </div>
        </footer>
    }
}
