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
                <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#d4a012" />
                    <stop offset="100%" stop-color="#b8860b" />
                </linearGradient>
                <linearGradient id="goldFill" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#d4a012" stop-opacity="0.25" />
                    <stop offset="100%" stop-color="#b8860b" stop-opacity="0.15" />
                </linearGradient>
            </defs>

            <circle
                cx=center
                cy=center
                r=radius
                fill="#f9f5ed"
                stroke="#e7e5e4"
                stroke-width="1"
            />

            {grid_levels.iter().map(|&scale| {
                let r = radius * scale;
                view! {
                    <circle
                        cx=center
                        cy=center
                        r=r
                        fill="none"
                        stroke="#d6d3d1"
                        stroke-width="1"
                        stroke-dasharray="3 3"
                    />
                }
            }).collect_view()}

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
                            stroke="#d6d3d1"
                            stroke-width="1"
                        />
                    }
                }).collect_view()
            }}

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
                        fill="url(#goldFill)"
                        stroke="url(#goldGradient)"
                        stroke-width="2.5"
                        stroke-linejoin="round"
                    />
                }
            }}

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
                            fill="#d4a012"
                            stroke="white"
                            stroke-width="2"
                        />
                    }
                }).collect_view()
            }}

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
                            fill="#78716c"
                            font-family="DM Sans, system-ui, sans-serif"
                            font-weight="500"
                        >
                            {point.label.clone()}
                        </text>
                    }
                }).collect_view()
            }}

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
                        fill="white"
                        stroke="#e7e5e4"
                        stroke-width="1"
                    />
                    <text
                        x=center
                        y=center - 6.0
                        text-anchor="middle"
                        font-size="22"
                        font-weight="600"
                        fill="#1c1917"
                        font-family="Cormorant Garamond, Georgia, serif"
                    >
                        {score}
                    </text>
                    <text
                        x=center
                        y=center + 12.0
                        text-anchor="middle"
                        font-size="9"
                        fill="#a8a29e"
                        font-family="DM Sans, system-ui, sans-serif"
                        text-transform="uppercase"
                        letter-spacing="0.05em"
                    >
                        "OVERALL"
                    </text>
                }
            }}
        </svg>
    }
}
