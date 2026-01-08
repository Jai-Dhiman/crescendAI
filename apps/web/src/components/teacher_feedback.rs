use leptos::prelude::*;

#[component]
pub fn TeacherFeedback(#[prop(into)] feedback: String) -> impl IntoView {
    view! {
        <div class="card p-6">
            <div class="flex items-center gap-3 mb-5 pb-5 border-b border-stone-100">
                <div class="w-10 h-10 rounded-lg bg-gold-100 flex items-center justify-center">
                    <svg class="w-5 h-5 text-gold-600" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
                    </svg>
                </div>
                <div>
                    <h3 class="font-display text-heading-lg text-stone-900">
                        "Teacher Feedback"
                    </h3>
                    <p class="text-label-sm text-stone-400 uppercase tracking-wide">
                        "Personalized guidance"
                    </p>
                </div>
            </div>

            <div class="text-body-md text-stone-600 leading-relaxed space-y-4">
                {feedback.split("\n\n").map(|paragraph| {
                    view! { <p>{paragraph.to_string()}</p> }
                }).collect_view()}
            </div>
        </div>
    }
}
