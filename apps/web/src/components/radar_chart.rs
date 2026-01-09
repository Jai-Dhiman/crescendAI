use leptos::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct RadarDataPoint {
    pub label: String,
    pub value: f64,
}

#[component]
pub fn RadarChart(
    #[prop(into)] data: Signal<Vec<RadarDataPoint>>,
    #[prop(default = 400)] size: u32,
) -> impl IntoView {
    let center = size as f64 / 2.0;
    let radius = center * 0.62;
    let label_radius = center * 0.88;

    let grid_levels = vec![0.25, 0.5, 0.75, 1.0];

    view! {
        <svg
            width=size
            height=size
            viewBox=format!("0 0 {} {}", size, size)
            class="radar-chart"
            role="img"
            aria-label="Radar chart showing performance analysis across 19 dimensions"
        >
            <defs>
                // Sepia gradient for the data polygon stroke
                <linearGradient id="sepiaGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#a69276" />
                    <stop offset="100%" stop-color="#8b7355" />
                </linearGradient>
                // Sepia fill with transparency
                <linearGradient id="sepiaFill" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#a69276" stop-opacity="0.25" />
                    <stop offset="100%" stop-color="#8b7355" stop-opacity="0.15" />
                </linearGradient>
            </defs>

            // Background circle (parchment color)
            <circle
                cx=center
                cy=center
                r=radius
                fill="#f5f1e8"
                stroke="#ede6d9"
                stroke-width="1"
            />

            // Grid circles
            {grid_levels.iter().map(|&scale| {
                let r = radius * scale;
                view! {
                    <circle
                        cx=center
                        cy=center
                        r=r
                        fill="none"
                        stroke="#e0d5c3"
                        stroke-width="1"
                        stroke-dasharray="3 3"
                    />
                }
            }).collect_view()}

            // Axis lines from center to edge
            {move || {
                let points = data.get();
                let n = points.len();
                points.iter().enumerate().map(|(i, _)| {
                    let angle = (2.0 * PI * i as f64 / n as f64) - PI / 2.0;
                    let x2 = center + radius * angle.cos();
                    let y2 = center + radius * angle.sin();
                    view! {
                        <line
                            x1=center
                            y1=center
                            x2=x2
                            y2=y2
                            stroke="#e0d5c3"
                            stroke-width="1"
                        />
                    }
                }).collect_view()
            }}

            // Data polygon
            {move || {
                let points = data.get();
                let n = points.len();
                let polygon_points: String = points.iter().enumerate().map(|(i, point)| {
                    let angle = (2.0 * PI * i as f64 / n as f64) - PI / 2.0;
                    let r = radius * point.value;
                    let x = center + r * angle.cos();
                    let y = center + r * angle.sin();
                    format!("{:.1},{:.1}", x, y)
                }).collect::<Vec<_>>().join(" ");

                view! {
                    <polygon
                        points=polygon_points
                        fill="url(#sepiaFill)"
                        stroke="url(#sepiaGradient)"
                        stroke-width="2.5"
                        stroke-linejoin="round"
                    />
                }
            }}

            // Data points (dots at each vertex)
            {move || {
                let points = data.get();
                let n = points.len();
                points.iter().enumerate().map(|(i, point)| {
                    let angle = (2.0 * PI * i as f64 / n as f64) - PI / 2.0;
                    let r = radius * point.value;
                    let x = center + r * angle.cos();
                    let y = center + r * angle.sin();
                    view! {
                        <circle
                            cx=x
                            cy=y
                            r="5"
                            fill="#a69276"
                            stroke="#fefdfb"
                            stroke-width="2"
                        />
                    }
                }).collect_view()
            }}

            // Labels
            {move || {
                let points = data.get();
                let n = points.len();
                points.iter().enumerate().map(|(i, point)| {
                    let angle = (2.0 * PI * i as f64 / n as f64) - PI / 2.0;
                    let x = center + label_radius * angle.cos();
                    let y = center + label_radius * angle.sin();

                    let anchor = if angle.cos() < -0.1 {
                        "end"
                    } else if angle.cos() > 0.1 {
                        "start"
                    } else {
                        "middle"
                    };

                    let dy = if angle.sin() < -0.5 {
                        "-0.3em"
                    } else if angle.sin() > 0.5 {
                        "1em"
                    } else {
                        "0.35em"
                    };

                    view! {
                        <text
                            x=x
                            y=y
                            text-anchor=anchor
                            dominant-baseline="middle"
                            dy=dy
                            font-size="10"
                            fill="#746a5c"
                            font-family="Inter, system-ui, sans-serif"
                            font-weight="500"
                        >
                            {point.label.clone()}
                        </text>
                    }
                }).collect_view()
            }}

            // Center score display
            {move || {
                let points = data.get();
                let avg: f64 = if points.is_empty() {
                    0.0
                } else {
                    points.iter().map(|p| p.value).sum::<f64>() / points.len() as f64
                };
                let score = (avg * 100.0).round() as u32;

                view! {
                    <circle
                        cx=center
                        cy=center
                        r="32"
                        fill="#fefdfb"
                        stroke="#ede6d9"
                        stroke-width="1"
                    />
                    <text
                        x=center
                        y=center - 6.0
                        text-anchor="middle"
                        font-size="22"
                        font-weight="500"
                        fill="#2d2926"
                        font-family="Cormorant Garamond, Georgia, serif"
                    >
                        {score}
                    </text>
                    <text
                        x=center
                        y=center + 12.0
                        text-anchor="middle"
                        font-size="9"
                        fill="#968c7d"
                        font-family="Inter, system-ui, sans-serif"
                        text-transform="uppercase"
                        letter-spacing="0.1em"
                    >
                        "OVERALL"
                    </text>
                }
            }}
        </svg>
    }
}

#[component]
pub fn CollapsibleRadarChart(
    #[prop(into)] data: Signal<Vec<RadarDataPoint>>,
    #[prop(default = 400)] size: u32,
) -> impl IntoView {
    let (is_expanded, set_expanded) = signal(true); // Default to expanded

    view! {
        <div class="card overflow-hidden">
            <button
                on:click=move |_| set_expanded.update(|v| *v = !*v)
                class="w-full px-6 py-4 flex items-center justify-between text-left
                       hover:bg-paper-100 transition-colors"
            >
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-lg bg-sepia-100 flex items-center justify-center">
                        <svg class="w-4 h-4 text-sepia-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <div>
                        <span class="block font-display text-heading-md text-ink-800">
                            "Dimension Details"
                        </span>
                        <span class="block text-label-sm text-ink-400">
                            "19-dimension radar visualization"
                        </span>
                    </div>
                </div>
                <svg
                    class=move || format!(
                        "w-5 h-5 text-ink-400 transition-transform duration-200 {}",
                        if is_expanded.get() { "rotate-180" } else { "" }
                    )
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    viewBox="0 0 24 24"
                >
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            <div
                class=move || if is_expanded.get() {
                    "max-h-[600px] opacity-100 transition-all duration-300"
                } else {
                    "max-h-0 opacity-0 overflow-hidden transition-all duration-200"
                }
            >
                <div class="p-8 border-t border-paper-200 flex justify-center">
                    <RadarChart data=data size=size />
                </div>
            </div>
        </div>
    }
}
