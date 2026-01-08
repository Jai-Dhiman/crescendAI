use leptos::prelude::*;
use leptos_meta::*;

use crate::App;

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone()/>
                <HydrationScripts options=options.clone()/>
                <MetaTags/>
            </head>
            <body class="bg-cream-50 text-stone-800 min-h-screen antialiased">
                <App/>
            </body>
        </html>
    }
}
