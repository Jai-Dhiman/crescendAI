use worker::Env;

#[derive(Clone)]
pub struct AppState {
    pub env: Env,
}

impl AppState {
    pub fn new(env: Env) -> Self {
        Self { env }
    }
}
