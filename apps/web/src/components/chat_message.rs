use leptos::prelude::*;
use std::collections::HashMap;

use crate::components::expandable_citation::{CitationExpansionState, ExpandableCitation};
use crate::models::Citation;

/// Role of the message sender
#[derive(Clone, Debug, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
}

/// A chat message with optional citations
#[derive(Clone, Debug)]
pub struct ChatMessageData {
    pub role: MessageRole,
    pub content: String,
    pub citations: Vec<Citation>,
}

/// Individual chat message component
#[component]
pub fn ChatMessage(message: ChatMessageData) -> impl IntoView {
    let is_user = message.role == MessageRole::User;

    view! {
        <div class=format!(
            "flex {}",
            if is_user { "justify-end" } else { "justify-start" }
        )>
            <div class=format!(
                "max-w-[85%] rounded-2xl px-4 py-3 {}",
                if is_user {
                    "bg-sepia-600 text-white rounded-br-sm"
                } else {
                    "bg-paper-100 text-ink-700 rounded-bl-sm border border-paper-200"
                }
            )>
                {if is_user {
                    view! {
                        <p class="text-body-md">{message.content.clone()}</p>
                    }.into_any()
                } else {
                    view! {
                        <AssistantMessage
                            content=message.content.clone()
                            citations=message.citations.clone()
                        />
                    }.into_any()
                }}
            </div>
        </div>
    }
}

/// Assistant message with expandable citations
#[component]
fn AssistantMessage(content: String, citations: Vec<Citation>) -> impl IntoView {
    let expansion_state = CitationExpansionState::new();

    // Build citation map
    let citation_map: HashMap<i32, Citation> = citations
        .iter()
        .map(|c| (c.number, c.clone()))
        .collect();

    // Parse text into segments
    let segments = parse_text_with_citations(&content);

    view! {
        <div class="text-body-md font-serif leading-relaxed">
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
                            view! {
                                <span class="inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-paper-200 text-ink-500 rounded">
                                    {format!("[{}]", number)}
                                </span>
                            }.into_any()
                        }
                    }
                }
            }).collect_view()}
        </div>
    }
}

/// Text segment types for parsing
enum TextSegment {
    Plain(String),
    Citation(i32),
}

/// Parse text into segments of plain text and citations
fn parse_text_with_citations(text: &str) -> Vec<TextSegment> {
    use regex::Regex;

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

/// Typing indicator for when AI is responding
#[component]
pub fn TypingIndicator() -> impl IntoView {
    view! {
        <div class="flex justify-start">
            <div class="bg-paper-100 rounded-2xl rounded-bl-sm px-4 py-3 border border-paper-200">
                <div class="flex items-center gap-1.5">
                    <div class="w-2 h-2 bg-sepia-400 rounded-full animate-bounce" style="animation-delay: 0ms" />
                    <div class="w-2 h-2 bg-sepia-400 rounded-full animate-bounce" style="animation-delay: 150ms" />
                    <div class="w-2 h-2 bg-sepia-400 rounded-full animate-bounce" style="animation-delay: 300ms" />
                </div>
            </div>
        </div>
    }
}
