#![cfg(feature = "hydrate")]

use leptos::prelude::*;
use wasm_bindgen::prelude::wasm_bindgen;

use crescendai::App;

#[wasm_bindgen(start)]
pub fn hydrate() {
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}
