use leptos::prelude::*;
use regex::Regex;
use std::collections::HashMap;

use crate::components::expandable_citation::{CitationExpansionState, ExpandableCitation};
use crate::models::{CitedFeedback, Citation, SourceType};

#[component]
pub fn TeacherFeedback(feedback: CitedFeedback) -> impl IntoView {
    let has_citations = !feedback.citations.is_empty();
    let citations = feedback.citations.clone();
    let citations_for_footer = citations.clone();

    // Create expansion state for this feedback
    let expansion_state = CitationExpansionState::new();

    // Build a map from citation number to citation for quick lookup
    let citation_map: HashMap<i32, Citation> = citations
        .iter()
        .map(|c| (c.number, c.clone()))
        .collect();

    view! {
        <div class="card p-6">
            <div class="flex items-center gap-3 mb-5 pb-5 border-b border-paper-200">
                <div class="w-10 h-10 rounded-lg bg-sepia-100 flex items-center justify-center">
                    <svg class="w-5 h-5 text-sepia-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
                    </svg>
                </div>
                <div>
                    <h3 class="font-display text-heading-lg text-ink-800">
                        "AI Teacher Feedback"
                    </h3>
                    <p class="text-label-sm text-sepia-500 uppercase tracking-wider">
                        {if has_citations { "Grounded in Pedagogy Sources" } else { "Example Application" }}
                    </p>
                </div>
            </div>

            <div class="text-body-md text-ink-600 leading-relaxed space-y-4 font-serif feedback-content">
                {feedback.plain_text.split("\n\n").map(|paragraph| {
                    let citation_map = citation_map.clone();
                    let expansion_state = expansion_state.clone();
                    view! {
                        <FeedbackParagraph
                            text=paragraph.to_string()
                            citation_map=citation_map
                            expansion_state=expansion_state
                        />
                    }
                }).collect_view()}
            </div>

            // Collapsible sources footer
            {has_citations.then(|| {
                view! {
                    <SourcesFooter citations=citations_for_footer />
                }
            })}
        </div>
    }
}

/// A paragraph that renders text with interactive citation markers
#[component]
fn FeedbackParagraph(
    text: String,
    citation_map: HashMap<i32, Citation>,
    expansion_state: CitationExpansionState,
) -> impl IntoView {
    // Parse the text into segments (text and citations)
    let segments = parse_text_with_citations(&text);

    view! {
        <p>
            {segments.into_iter().map(|segment| {
                match segment {
                    TextSegment::Plain(text) => {
                        view! { <span>{text}</span> }.into_any()
                    }
                    TextSegment::Citation(number) => {
                        if let Some(citation) = citation_map.get(&number) {
                            let citation = citation.clone();
                            let state = expansion_state.clone();
                            let expanded = state.is_expanded(number);
                            let on_toggle = move || state.toggle(number);

                            view! {
                                <ExpandableCitation
                                    citation=citation
                                    expanded=expanded
                                    on_toggle=on_toggle
                                />
                            }.into_any()
                        } else {
                            // Fallback for unknown citation numbers
                            view! {
                                <span class="citation-marker inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-stone-100 text-stone-500 rounded">
                                    {format!("[{}]", number)}
                                </span>
                            }.into_any()
                        }
                    }
                }
            }).collect_view()}
        </p>
    }
}

/// Text segment types for parsing
enum TextSegment {
    Plain(String),
    Citation(i32),
}

/// Parse text into segments of plain text and citations
fn parse_text_with_citations(text: &str) -> Vec<TextSegment> {
    let citation_regex = Regex::new(r"\[(\d+)\]").unwrap();
    let mut segments = Vec::new();
    let mut last_end = 0;

    for cap in citation_regex.captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let start = full_match.start();
        let end = full_match.end();

        // Add preceding plain text if any
        if start > last_end {
            segments.push(TextSegment::Plain(text[last_end..start].to_string()));
        }

        // Add citation
        if let Ok(number) = cap[1].parse::<i32>() {
            segments.push(TextSegment::Citation(number));
        }

        last_end = end;
    }

    // Add remaining text
    if last_end < text.len() {
        segments.push(TextSegment::Plain(text[last_end..].to_string()));
    }

    segments
}

/// Collapsible sources footer
#[component]
fn SourcesFooter(citations: Vec<Citation>) -> impl IntoView {
    let (collapsed, set_collapsed) = signal(false);

    view! {
        <div class="mt-6 pt-4 border-t border-paper-200">
            <button
                type="button"
                class="flex items-center gap-2 text-label-sm text-sepia-600 uppercase tracking-wider mb-3 hover:text-sepia-800 transition-colors"
                on:click=move |_| set_collapsed.update(|c| *c = !*c)
            >
                <svg
                    class="w-4 h-4 transition-transform duration-200"
                    class:rotate-180=move || !collapsed.get()
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    viewBox="0 0 24 24"
                >
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
                "Sources"
                <span class="text-sepia-400 normal-case">
                    {format!("({})", citations.len())}
                </span>
            </button>

            <Show when=move || !collapsed.get()>
                <ul class="space-y-2 text-body-sm text-ink-500 animate-in slide-in-from-top-2 duration-200">
                    {citations.iter().map(|citation| {
                        view! { <CitationFootnote citation={citation.clone()} /> }
                    }).collect_view()}
                </ul>
            </Show>
        </div>
    }
}

#[component]
fn CitationFootnote(citation: Citation) -> impl IntoView {
    let footnote_text = citation.format_footnote();
    let url = citation.get_timestamped_url();

    let icon = match citation.source_type {
        SourceType::Book => view! {
            <svg class="w-4 h-4 text-sepia-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
        }.into_any(),
        SourceType::Masterclass => view! {
            <svg class="w-4 h-4 text-sepia-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
            </svg>
        }.into_any(),
        SourceType::Letter | SourceType::Journal => view! {
            <svg class="w-4 h-4 text-sepia-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
        }.into_any(),
    };

    view! {
        <li class="flex items-start gap-2">
            {icon}
            {match url {
                Some(link) => view! {
                    <a
                        href={link}
                        target="_blank"
                        rel="noopener noreferrer"
                        class="hover:text-sepia-600 hover:underline transition-colors"
                    >
                        {footnote_text}
                    </a>
                }.into_any(),
                None => view! {
                    <span>{footnote_text}</span>
                }.into_any(),
            }}
        </li>
    }
}
