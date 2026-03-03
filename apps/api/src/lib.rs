#![recursion_limit = "512"]

pub mod app;
pub mod components;
pub mod models;
pub mod pages;

#[cfg(feature = "ssr")]
pub mod api;
#[cfg(feature = "ssr")]
pub mod services;
#[cfg(feature = "ssr")]
pub mod state;
#[cfg(feature = "ssr")]
pub mod shell;
#[cfg(feature = "ssr")]
pub mod server;

// Re-export for convenience
pub use app::App;

// WASM hydration entry point
#[cfg(feature = "hydrate")]
mod hydrate {
    use wasm_bindgen::prelude::wasm_bindgen;
    use crate::App;

    #[wasm_bindgen(start)]
    pub fn hydrate() {
        console_error_panic_hook::set_once();
        leptos::mount::hydrate_body(App);
    }
}
