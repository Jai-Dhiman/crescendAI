use axum::extract::FromRef;
use leptos::config::LeptosOptions;
use worker::send::SendWrapper;
use worker::Env;

#[derive(Clone)]
pub struct AppState {
    pub leptos_options: LeptosOptions,
    pub env: SendWrapper<Env>,
}

impl AppState {
    pub fn new(env: Env, leptos_options: LeptosOptions) -> Self {
        Self {
            leptos_options,
            env: SendWrapper::new(env),
        }
    }

    #[allow(dead_code)]
    pub fn kv(&self, binding: &str) -> worker::Result<worker::kv::KvStore> {
        self.env.kv(binding)
    }

    #[allow(dead_code)]
    pub fn bucket(&self, binding: &str) -> worker::Result<worker::Bucket> {
        self.env.bucket(binding)
    }

    #[allow(dead_code)]
    pub fn d1(&self, binding: &str) -> worker::Result<worker::d1::D1Database> {
        self.env.d1(binding)
    }

    #[allow(dead_code)]
    pub fn ai(&self, binding: &str) -> worker::Result<worker::Ai> {
        self.env.ai(binding)
    }

    /// Get a generic binding by name.
    /// Use this for bindings without native Rust support (e.g., Vectorize).
    /// Returns the binding as the specified type.
    #[allow(dead_code)]
    pub fn get_binding<T: worker::EnvBinding>(&self, binding: &str) -> worker::Result<T> {
        self.env.get_binding(binding)
    }
}

impl FromRef<AppState> for LeptosOptions {
    fn from_ref(state: &AppState) -> Self {
        state.leptos_options.clone()
    }
}
