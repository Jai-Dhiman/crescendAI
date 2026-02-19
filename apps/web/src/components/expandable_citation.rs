use leptos::prelude::*;

use crate::models::{Citation, SourceType};

/// A clickable citation marker that expands inline to show source details
#[component]
pub fn ExpandableCitation(
    citation: Citation,
    /// Whether this citation is currently expanded
    #[prop(into)]
    expanded: Signal<bool>,
    /// Callback when clicked to toggle expansion
    on_toggle: impl Fn() + Send + Sync + 'static + Clone,
) -> impl IntoView {
    let citation_for_view = citation.clone();
    let on_toggle_click = on_toggle.clone();

    view! {
        <span class="inline">
            <button
                type="button"
                class="citation-marker inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-clay-100 text-clay-700 rounded cursor-pointer hover:bg-clay-200 hover:scale-110 transition-all duration-150 align-baseline mx-0.5"
                on:click=move |_| on_toggle_click()
                aria-expanded=move || expanded.get().to_string()
                aria-label=format!("Citation {}: {}", citation.number, citation.title)
            >
                {format!("[{}]", citation.number)}
            </button>

            // Inline expansion appears right after the marker
            <Show when=move || expanded.get()>
                <CitationExpansion
                    citation=citation_for_view.clone()
                    on_close=on_toggle.clone()
                />
            </Show>
        </span>
    }
}

/// The expanded citation box showing source details
#[component]
fn CitationExpansion(
    citation: Citation,
    on_close: impl Fn() + Send + Sync + 'static,
) -> impl IntoView {
    let url = citation.get_timestamped_url();
    let footnote = citation.format_footnote();

    let icon = match citation.source_type {
        SourceType::Book => view! {
            <svg class="w-4 h-4 text-clay-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
        }.into_any(),
        SourceType::Masterclass => view! {
            <svg class="w-4 h-4 text-clay-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
            </svg>
        }.into_any(),
        SourceType::Letter | SourceType::Journal => view! {
            <svg class="w-4 h-4 text-clay-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
        }.into_any(),
    };

    view! {
        <div class="citation-expansion block mt-2 mb-3 p-3 bg-paper-50 border border-clay-200 rounded-lg shadow-sm animate-in slide-in-from-top-2 duration-200">
            // Header with source info and close button
            <div class="flex items-start justify-between gap-2 mb-2">
                <div class="flex items-start gap-2 min-w-0">
                    {icon}
                    <div class="min-w-0">
                        <p class="text-label-sm text-clay-600 font-medium truncate">
                            {footnote}
                        </p>
                    </div>
                </div>
                <button
                    type="button"
                    class="text-clay-400 hover:text-clay-600 transition-colors flex-shrink-0 p-0.5"
                    on:click=move |_| on_close()
                    aria-label="Close citation"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            // Quote text
            {citation.quote.map(|quote| {
                view! {
                    <blockquote class="text-body-sm text-ink-500 italic border-l-2 border-clay-200 pl-3 my-2">
                        {format!("\"{}\"", quote)}
                    </blockquote>
                }
            })}

            // Link to source
            {url.map(|link| {
                view! {
                    <a
                        href={link}
                        target="_blank"
                        rel="noopener noreferrer"
                        class="inline-flex items-center gap-1 text-label-sm text-clay-600 hover:text-clay-800 hover:underline transition-colors"
                    >
                        "View source"
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                        </svg>
                    </a>
                }
            })}
        </div>
    }
}

/// State manager for tracking which citations are expanded
#[derive(Clone, Default)]
pub struct CitationExpansionState {
    expanded_numbers: RwSignal<Vec<i32>>,
}

impl CitationExpansionState {
    pub fn new() -> Self {
        Self {
            expanded_numbers: RwSignal::new(vec![]),
        }
    }

    pub fn is_expanded(&self, number: i32) -> Signal<bool> {
        let expanded = self.expanded_numbers;
        Signal::derive(move || expanded.get().contains(&number))
    }

    pub fn toggle(&self, number: i32) {
        self.expanded_numbers.update(|nums| {
            if nums.contains(&number) {
                nums.retain(|&n| n != number);
            } else {
                // Close others and open this one (accordion behavior)
                nums.clear();
                nums.push(number);
            }
        });
    }

    pub fn close_all(&self) {
        self.expanded_numbers.set(vec![]);
    }
}
