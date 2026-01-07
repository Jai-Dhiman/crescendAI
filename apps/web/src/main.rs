use leptos::mount::mount_to_body;
use piano_feedback_web::app::App;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("Failed to initialize logger");

    mount_to_body(App);
}
