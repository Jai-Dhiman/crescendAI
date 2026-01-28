use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;

use crate::components::Header;
use crate::pages::{DemoPage, LandingPage};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/output.css"/>

        <Link rel="icon" type_="image/png" href="/crescendai.png"/>
        <Link rel="preconnect" href="https://fonts.googleapis.com"/>
        <Link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous"/>
        <Link
            href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap"
            rel="stylesheet"
        />

        <Title text="CrescendAI - Perceptual Piano Performance Analysis"/>
        <Meta name="description" content="Research demonstration exploring how audio-based deep learning models predict human perceptual ratings of piano performances across 19 musical dimensions."/>

        <Router>
            <div class="min-h-screen bg-gradient-page flex flex-col texture-paper">
                <Header />
                <main class="flex-1">
                    <Routes fallback=|| view! { <NotFound /> }>
                        <Route path=path!("/") view=LandingPage />
                        <Route path=path!("/demo") view=DemoPage />
                        <Route path=path!("/demo/:id") view=DemoPage />
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
        <div class="container-narrow text-center py-24 animate-fade-in">
            <div class="font-display text-[8rem] font-medium text-paper-300 leading-none mb-4">
                "404"
            </div>
            <h1 class="font-display text-display-md text-ink-900 mb-3">
                "Page Not Found"
            </h1>
            <p class="text-body-md text-ink-500 mb-8">
                "The page you're looking for doesn't exist or has been moved."
            </p>
            <a href="/" class="btn-primary">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/>
                </svg>
                "Return Home"
            </a>
        </div>
    }
}

#[component]
fn Footer() -> impl IntoView {
    view! {
        <footer class="border-t border-paper-300 mt-auto bg-paper-100">
            <div class="container-wide py-10">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    // Logo and name
                    <div class="flex items-center gap-3">
                        <img
                            src="/crescendai.png"
                            alt="CrescendAI Logo"
                            class="w-8 h-8 rounded-md"
                        />
                        <div>
                            <span class="font-display text-lg font-medium text-ink-800 block">
                                "CrescendAI"
                            </span>
                            <span class="text-label-sm text-ink-400">
                                "Research Project"
                            </span>
                        </div>
                    </div>

                    // Attribution
                    <p class="text-body-sm text-ink-500 text-center md:text-left">
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="text-sepia-600 hover:text-sepia-700 underline underline-offset-2">
                            "Paper"
                        </a>
                        " | Built on "
                        <a href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noopener" class="text-sepia-600 hover:text-sepia-700 underline underline-offset-2">
                            "PercePiano"
                        </a>
                        " dataset and "
                        <a href="https://github.com/tencent-ailab/MuQ" target="_blank" rel="noopener" class="text-sepia-600 hover:text-sepia-700 underline underline-offset-2">
                            "MuQ"
                        </a>
                        " audio encoder"
                    </p>

                    // Copyright
                    <p class="text-label-sm text-ink-400 uppercase tracking-wider">
                        "2026 Research Demo"
                    </p>
                </div>
            </div>
        </footer>
    }
}
