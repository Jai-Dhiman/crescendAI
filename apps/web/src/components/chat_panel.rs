use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::components::chat_input::ChatInput;
use crate::components::chat_message::{ChatMessage, ChatMessageData, MessageRole, TypingIndicator};
use crate::models::Citation;

/// Request payload for the chat API
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    pub performance_id: String,
    pub question: String,
}

/// Response from the chat API
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatResponse {
    pub answer: String,
    pub citations: Vec<Citation>,
}

/// Chat panel for Q&A about the performance
#[component]
pub fn ChatPanel(
    /// Performance ID for context
    performance_id: String,
) -> impl IntoView {
    let (messages, set_messages) = signal::<Vec<ChatMessageData>>(vec![]);
    let (is_loading, set_is_loading) = signal(false);
    let performance_id_for_submit = performance_id.clone();

    // Handle submitting a question
    let on_submit = move |question: String| {
        let performance_id = performance_id_for_submit.clone();

        // Add user message
        set_messages.update(|msgs| {
            msgs.push(ChatMessageData {
                role: MessageRole::User,
                content: question.clone(),
                citations: vec![],
            });
        });

        set_is_loading.set(true);

        // Make API call
        spawn_local(async move {
            match fetch_chat_response(&performance_id, &question).await {
                Ok(response) => {
                    set_messages.update(|msgs| {
                        msgs.push(ChatMessageData {
                            role: MessageRole::Assistant,
                            content: response.answer,
                            citations: response.citations,
                        });
                    });
                }
                Err(e) => {
                    set_messages.update(|msgs| {
                        msgs.push(ChatMessageData {
                            role: MessageRole::Assistant,
                            content: format!("Sorry, I couldn't process your question. Please try again. ({})", e),
                            citations: vec![],
                        });
                    });
                }
            }
            set_is_loading.set(false);
        });
    };

    view! {
        <div class="card flex flex-col h-[400px]">
            // Header
            <div class="flex items-center gap-3 p-4 border-b border-paper-200">
                <div class="w-8 h-8 rounded-lg bg-clay-100 flex items-center justify-center">
                    <svg class="w-4 h-4 text-clay-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
                    </svg>
                </div>
                <div>
                    <h3 class="font-display text-heading-md text-ink-800">
                        "Ask Your Teacher"
                    </h3>
                    <p class="text-label-sm text-clay-500">
                        "Get personalized advice grounded in pedagogy"
                    </p>
                </div>
            </div>

            // Messages area
            <div class="flex-1 overflow-y-auto p-4 space-y-4">
                {move || {
                    let msgs = messages.get();
                    if msgs.is_empty() {
                        view! {
                            <EmptyState />
                        }.into_any()
                    } else {
                        msgs.into_iter().map(|msg| {
                            view! { <ChatMessage message=msg /> }
                        }).collect_view().into_any()
                    }
                }}

                // Typing indicator
                <Show when=move || is_loading.get()>
                    <TypingIndicator />
                </Show>
            </div>

            // Input area
            <ChatInput
                on_submit=on_submit
                disabled=is_loading.get()
            />
        </div>
    }
}

/// Empty state when no messages yet
#[component]
fn EmptyState() -> impl IntoView {
    view! {
        <div class="flex flex-col items-center justify-center h-full text-center py-8">
            <div class="w-12 h-12 rounded-full bg-clay-50 flex items-center justify-center mb-4">
                <svg class="w-6 h-6 text-clay-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
                </svg>
            </div>
            <p class="text-body-md text-ink-500 mb-2">
                "Have questions about your performance?"
            </p>
            <p class="text-body-sm text-ink-400 max-w-xs">
                "Ask about technique, interpretation, or how to improve specific aspects of your playing."
            </p>
            <div class="mt-4 flex flex-wrap justify-center gap-2">
                <SuggestionChip text="How can I improve my pedaling?" />
                <SuggestionChip text="Tips for better legato?" />
                <SuggestionChip text="Chopin's rubato style" />
            </div>
        </div>
    }
}

/// Suggestion chip for quick questions
#[component]
fn SuggestionChip(text: &'static str) -> impl IntoView {
    view! {
        <span class="px-3 py-1.5 text-label-sm text-clay-600 bg-clay-50 rounded-full border border-clay-200 hover:bg-clay-100 cursor-pointer transition-colors">
            {text}
        </span>
    }
}

/// Fetch chat response from API
#[cfg(feature = "hydrate")]
async fn fetch_chat_response(performance_id: &str, question: &str) -> Result<ChatResponse, String> {
    use gloo_net::http::Request;

    let request = ChatRequest {
        performance_id: performance_id.to_string(),
        question: question.to_string(),
    };

    let response = Request::post("/api/chat")
        .json(&request)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if response.ok() {
        response
            .json::<ChatResponse>()
            .await
            .map_err(|e| e.to_string())
    } else {
        Err(format!("API error: {}", response.status()))
    }
}

/// SSR fallback - returns error since API calls happen client-side
#[cfg(not(feature = "hydrate"))]
async fn fetch_chat_response(_performance_id: &str, _question: &str) -> Result<ChatResponse, String> {
    Err("Chat is only available in the browser".to_string())
}
