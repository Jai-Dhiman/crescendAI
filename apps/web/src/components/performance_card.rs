use leptos::prelude::*;
use crate::models::Performance;

#[component]
pub fn PerformanceCard(performance: Performance) -> impl IntoView {
    let href = format!("/performance/{}", performance.id);
    let duration_str = format_duration(performance.duration_seconds);

    view! {
        <a
            href=href
            class="group block bg-slate-800/50 rounded-xl overflow-hidden hover:bg-slate-800 transition-all duration-300 hover:scale-[1.02] hover:shadow-xl hover:shadow-rose-500/10 border border-white/5 hover:border-rose-500/20"
        >
            <div class="aspect-video relative overflow-hidden bg-gradient-to-br from-slate-700 to-slate-800">
                <div class="absolute inset-0 flex items-center justify-center">
                    <div class="w-16 h-16 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-rose-500/20 transition-colors">
                        <svg class="w-8 h-8 text-white/60 group-hover:text-rose-400 transition-colors" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M8 5v14l11-7z"/>
                        </svg>
                    </div>
                </div>
                <div class="absolute bottom-2 right-2 px-2 py-1 bg-black/60 rounded text-xs text-white/80">
                    {duration_str}
                </div>
                <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
                <div class="absolute bottom-3 left-3">
                    <p class="text-sm text-white/70">{performance.composer.clone()}</p>
                </div>
            </div>
            <div class="p-4">
                <h3 class="font-serif text-lg font-semibold text-white mb-1 line-clamp-2 group-hover:text-rose-100 transition-colors">
                    {performance.piece_title.clone()}
                </h3>
                <p class="text-sm text-rose-400">{performance.performer.clone()}</p>
                {performance.year_recorded.map(|year| view! {
                    <p class="text-xs text-white/40 mt-1">{format!("Recorded {}", year)}</p>
                })}
            </div>
        </a>
    }
}

fn format_duration(seconds: u32) -> String {
    let mins = seconds / 60;
    let secs = seconds % 60;
    format!("{}:{:02}", mins, secs)
}
