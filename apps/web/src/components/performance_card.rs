use leptos::prelude::*;
use crate::models::Performance;

#[component]
pub fn PerformanceCard(performance: Performance) -> impl IntoView {
    let href = format!("/performance/{}", performance.id);
    let duration_str = format_duration(performance.duration_seconds);

    view! {
        <a
            href=href
            class="group card-interactive overflow-hidden focus-visible:ring-2 focus-visible:ring-gold-500 focus-visible:ring-offset-2"
            aria-label=format!("Analyze {} by {}", performance.piece_title.clone(), performance.performer.clone())
        >
            // Thumbnail area
            <div class="aspect-video relative overflow-hidden bg-gradient-to-br from-stone-100 to-stone-200">
                // Play button overlay
                <div class="absolute inset-0 flex items-center justify-center">
                    <div class="w-14 h-14 rounded-full bg-white/90 shadow-elevation-3 flex items-center justify-center
                                transition-all duration-300 ease-out-expo
                                group-hover:bg-gradient-gold group-hover:scale-110 group-hover:shadow-gold">
                        <svg
                            class="w-6 h-6 text-stone-600 group-hover:text-white transition-colors ml-0.5"
                            fill="currentColor"
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                        >
                            <path d="M8 5v14l11-7z"/>
                        </svg>
                    </div>
                </div>

                // Duration badge
                <div class="absolute bottom-3 right-3 px-2 py-1 bg-stone-900/80 backdrop-blur-xs rounded text-label-sm text-white">
                    {duration_str}
                </div>

                // Composer overlay
                <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-stone-900/70 via-stone-900/30 to-transparent pt-12 pb-3 px-4">
                    <span class="text-label-md uppercase tracking-wider text-white/90">
                        {performance.composer.clone()}
                    </span>
                </div>
            </div>

            // Content area
            <div class="p-5">
                <h3 class="font-display text-heading-lg text-stone-900 mb-1.5 line-clamp-2
                           group-hover:text-gold-700 transition-colors duration-200">
                    {performance.piece_title.clone()}
                </h3>
                <p class="text-body-sm font-medium text-gold-600">
                    {performance.performer.clone()}
                </p>
                {performance.year_recorded.map(|year| view! {
                    <p class="text-label-sm text-stone-400 mt-2 uppercase tracking-wide">
                        {format!("Recorded {}", year)}
                    </p>
                })}

                // Analyze indicator
                <div class="mt-4 pt-4 border-t border-stone-100 flex items-center justify-between">
                    <span class="text-label-sm uppercase tracking-wider text-stone-400 group-hover:text-gold-600 transition-colors">
                        "Analyze Performance"
                    </span>
                    <svg
                        class="w-4 h-4 text-stone-300 group-hover:text-gold-500 group-hover:translate-x-1 transition-all duration-200"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        viewBox="0 0 24 24"
                        aria-hidden="true"
                    >
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                </div>
            </div>
        </a>
    }
}

fn format_duration(seconds: u32) -> String {
    let mins = seconds / 60;
    let secs = seconds % 60;
    format!("{}:{:02}", mins, secs)
}
