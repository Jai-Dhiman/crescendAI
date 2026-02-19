use leptos::prelude::*;
use crate::models::CategoryScore;

#[component]
pub fn CategoryCard(category: CategoryScore) -> impl IntoView {
    let bar_width = format!("{}%", (category.score * 100.0).min(100.0));
    let bar_color = match category.label.as_str() {
        "Strong" => "bg-clay-500",
        "Good" => "bg-clay-400",
        "Developing" => "bg-clay-300",
        _ => "bg-clay-200",
    };

    let name = category.name.clone();
    let label = category.label.clone();
    let summary = category.summary.clone();
    let practice_tip = category.practice_tip.clone();

    view! {
        <div class="card p-6">
            <div class="flex items-start gap-4">
                // Category icon
                <div class="flex-shrink-0 w-10 h-10 rounded-lg bg-clay-100 flex items-center justify-center text-clay-600">
                    <CategoryIcon name=category.name />
                </div>

                // Content
                <div class="flex-1 min-w-0">
                    // Header: name + label
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-display text-heading-sm text-ink-800">
                            {name}
                        </h3>
                        <span class="text-label-sm text-clay-600 font-medium">
                            {label}
                        </span>
                    </div>

                    // Score bar
                    <div class="h-1.5 bg-paper-200 rounded-full overflow-hidden mb-3">
                        <div
                            class=format!("h-full rounded-full transition-all duration-700 {}", bar_color)
                            style=format!("width: {}", bar_width)
                        />
                    </div>

                    // Summary
                    <p class="text-body-sm text-ink-600 mb-3">
                        {summary}
                    </p>

                    // Practice tip
                    <div class="bg-paper-100 rounded-md p-3 border border-paper-200">
                        <p class="text-label-sm text-clay-600 mb-1">"Practice tip"</p>
                        <p class="text-body-sm text-ink-600">{practice_tip}</p>
                    </div>
                </div>
            </div>
        </div>
    }
}

#[component]
fn CategoryIcon(#[prop(into)] name: String) -> impl IntoView {
    match name.as_str() {
        "Sound Quality" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M12 6v12M8 8v8M16 8v8M4 10v4M20 10v4"/>
            </svg>
        }.into_any(),
        "Musical Shaping" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M3 12c3-6 6-6 9 0s6 6 9 0"/>
            </svg>
        }.into_any(),
        "Technical Control" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="6" width="20" height="12" rx="1"/>
                <path d="M7 6v7M12 6v7M17 6v7M9.5 6v4M14.5 6v4"/>
            </svg>
        }.into_any(),
        "Interpretive Choices" => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <path d="M9 18V5l8-3v13"/>
                <circle cx="7" cy="18" r="2"/>
                <circle cx="15" cy="15" r="2"/>
            </svg>
        }.into_any(),
        _ => view! {
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
            </svg>
        }.into_any(),
    }
}
