use worker::*;

#[event(fetch)]
pub async fn main(_req: Request, _env: Env, _ctx: worker::Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    
    Response::from_json(&serde_json::json!({
        "status": "healthy",
        "message": "Minimal worker is running"
    }))
}
