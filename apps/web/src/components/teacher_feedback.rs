use leptos::prelude::*;

#[component]
pub fn TeacherFeedback(
    #[prop(into)] feedback: String,
) -> impl IntoView {
    view! {
        <div class="bg-slate-800/50 rounded-xl p-6 border border-white/5">
            <div class="flex items-center gap-3 mb-4">
                <div class="w-10 h-10 rounded-full bg-rose-500/20 flex items-center justify-center">
                    <svg class="w-5 h-5 text-rose-400" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 14l9-5-9-5-9 5 9 5z"/>
                        <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/>
                        <path fill-rule="evenodd" d="M12 14l-9-5v6.072a2 2 0 001.024 1.746l7.152 4.081a2 2 0 001.648 0l7.152-4.081A2 2 0 0021 15.072V9l-9 5z" clip-rule="evenodd"/>
                    </svg>
                </div>
                <h3 class="text-xl font-serif font-semibold text-white">"Teacher Feedback"</h3>
            </div>
            <div class="text-white/80 leading-relaxed whitespace-pre-wrap space-y-4">
                {feedback.split("\n\n").map(|paragraph| {
                    view! { <p>{paragraph.to_string()}</p> }
                }).collect_view()}
            </div>
        </div>
    }
}
