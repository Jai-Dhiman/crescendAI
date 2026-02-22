use leptos::prelude::*;
use leptos_meta::*;
use leptos_router::components::{Route, Router, Routes};
use leptos_router::hooks::use_params_map;
use leptos_router::path;

use crate::components::Header;
use crate::pages::{AnalyzePage, LandingPage};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/output.css"/>

        <Link rel="icon" type_="image/png" href="/crescendai.png"/>
        <Link rel="preconnect" href="https://fonts.googleapis.com"/>
        <Link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous"/>
        <Link
            href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap"
            rel="stylesheet"
        />

        <Title text="Crescend -- A Teacher for Every Pianist"/>
        <Meta name="description" content="Record yourself playing piano. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing."/>

        <Router>
            <div class="min-h-screen bg-paper-50 flex flex-col texture-grain">
                <Header />
                <main class="flex-1">
                    <Routes fallback=|| view! { <NotFound /> }>
                        <Route path=path!("/") view=LandingPage />
                        <Route path=path!("/analyze") view=AnalyzePage />
                        <Route path=path!("/analyze/:id") view=AnalyzePage />
                        <Route path=path!("/demo") view=DemoRedirect />
                        <Route path=path!("/demo/:id") view=DemoRedirect />
                    </Routes>
                </main>
                <Footer />
            </div>
        </Router>
    }
}

#[component]
fn DemoRedirect() -> impl IntoView {
    let params = use_params_map();

    #[cfg(feature = "hydrate")]
    {
        Effect::new(move |_| {
            if let Some(window) = web_sys::window() {
                let id = params.read().get("id");
                let target = match id {
                    Some(id) => format!("/analyze/{}", id),
                    None => "/analyze".to_string(),
                };
                let _ = window.location().replace(&target);
            }
        });
    }

    #[cfg(not(feature = "hydrate"))]
    {
        let _ = params;
    }

    view! {
        <div class="text-center py-20">
            <p class="text-body-md text-ink-500">"Redirecting..."</p>
        </div>
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
        <footer class="mt-auto py-12 lg:py-16">
            <div class="container-editorial">
                <hr class="editorial-rule mb-12 lg:mb-16" />

                <div class="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-8 items-start text-center lg:text-left">
                    // Left: large brand name
                    <a href="/" class="font-display text-display-md text-ink-900 tracking-tight">
                        "Crescend"
                    </a>

                    // Center: page links
                    <nav class="flex flex-row lg:flex-col items-center lg:items-start gap-3 justify-center text-body-sm">
                        <a href="https://arxiv.org/abs/2601.19029" target="_blank" rel="noopener" class="editorial-nav-link">
                            "Paper"
                        </a>
                        <a href="/#how-it-works" class="editorial-nav-link">
                            "How It Works"
                        </a>
                        <a href="/analyze" class="editorial-nav-link">
                            "Analyze"
                        </a>
                    </nav>

                    // Right: privacy + copyright
                    <div class="lg:text-right">
                        <p class="text-body-xs text-ink-500 mb-1">
                            "Your recordings are yours. We don't store or train on your data."
                        </p>
                        <p class="text-label-sm text-ink-400">
                            "2026 Crescend"
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    }
}
