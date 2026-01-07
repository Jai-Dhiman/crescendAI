use leptos::prelude::*;
use crate::models::PracticeTip;

#[component]
pub fn PracticeTips(
    #[prop(into)] tips: Vec<PracticeTip>,
) -> impl IntoView {
    view! {
        <div class="bg-slate-800/50 rounded-xl p-6 border border-white/5">
            <h3 class="text-xl font-serif font-semibold text-white mb-4">"Practice Insights"</h3>
            <ul class="space-y-4">
                {tips.into_iter().map(|tip| {
                    view! {
                        <li class="flex gap-3">
                            <div class="w-6 h-6 rounded-full bg-rose-500/20 flex-shrink-0 flex items-center justify-center mt-0.5">
                                <svg class="w-3 h-3 text-rose-400" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                                </svg>
                            </div>
                            <div>
                                <h4 class="font-semibold text-white">{tip.title}</h4>
                                <p class="text-white/60 text-sm mt-1">{tip.description}</p>
                            </div>
                        </li>
                    }
                }).collect_view()}
            </ul>
        </div>
    }
}
