use leptos::prelude::*;
use crate::models::PracticeTip;

#[component]
pub fn PracticeTips(
    #[prop(into)] tips: Vec<PracticeTip>,
) -> impl IntoView {
    view! {
        <div class="card p-6">
            // Header
            <div class="flex items-center gap-3 mb-5 pb-5 border-b border-stone-100">
                <div class="w-10 h-10 rounded-lg bg-gold-100 flex items-center justify-center">
                    <svg class="w-5 h-5 text-gold-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                    </svg>
                </div>
                <div>
                    <h3 class="font-display text-heading-lg text-stone-900">
                        "Practice Insights"
                    </h3>
                    <p class="text-label-sm text-stone-400 uppercase tracking-wide">
                        "Actionable suggestions"
                    </p>
                </div>
            </div>

            // Tips list
            <ul class="space-y-4" role="list">
                {tips.into_iter().enumerate().map(|(i, tip)| {
                    view! {
                        <li class="flex gap-4">
                            // Number indicator
                            <div class="flex-shrink-0 w-6 h-6 rounded-md bg-gold-100 flex items-center justify-center">
                                <span class="text-label-sm font-medium text-gold-700">
                                    {i + 1}
                                </span>
                            </div>
                            <div class="flex-1 min-w-0">
                                <h4 class="font-medium text-body-md text-stone-800 mb-0.5">
                                    {tip.title}
                                </h4>
                                <p class="text-body-sm text-stone-500">
                                    {tip.description}
                                </p>
                            </div>
                        </li>
                    }
                }).collect_view()}
            </ul>
        </div>
    }
}
