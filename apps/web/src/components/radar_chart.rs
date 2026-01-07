use leptos::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct RadarDataPoint {
    pub label: String,
    pub value: f64, // 0.0 to 1.0
}

#[component]
pub fn RadarChart(
    #[prop(into)] data: Signal<Vec<RadarDataPoint>>,
    #[prop(default = 400)] size: u32,
) -> impl IntoView {
    let center = size as f64 / 2.0;
    let radius = center * 0.65;
    let label_radius = center * 0.88;

    // Grid levels
    let grid_levels = vec![0.25, 0.5, 0.75, 1.0];

    view! {
        <svg
            width=size
            height=size
            viewBox=format!("0 0 {} {}", size, size)
            class="radar-chart"
        >
            // Background
            <circle
                cx=center
                cy=center
                r=radius
                fill="rgba(15, 23, 42, 0.8)"
                stroke="rgba(255,255,255,0.1)"
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
                        stroke="rgba(255,255,255,0.1)"
                        stroke-width="1"
                        stroke-dasharray="4 4"
                    />
                }
            }).collect_view()}

            // Axis lines
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
                            stroke="rgba(255,255,255,0.15)"
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
                        fill="rgba(244, 63, 94, 0.25)"
                        stroke="#f43f5e"
                        stroke-width="2"
                    />
                }
            }}

            // Data points
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
                            r="4"
                            fill="#f43f5e"
                            stroke="white"
                            stroke-width="1"
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

                    // Determine text anchor based on position
                    let anchor = if angle.cos() < -0.1 {
                        "end"
                    } else if angle.cos() > 0.1 {
                        "start"
                    } else {
                        "middle"
                    };

                    // Adjust y for top/bottom labels
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
                            fill="rgba(255,255,255,0.6)"
                            font-family="Inter, system-ui, sans-serif"
                        >
                            {point.label.clone()}
                        </text>
                    }
                }).collect_view()
            }}

            // Center score
            {move || {
                let points = data.get();
                let avg: f64 = if points.is_empty() {
                    0.0
                } else {
                    points.iter().map(|p| p.value).sum::<f64>() / points.len() as f64
                };
                let score = (avg * 100.0).round() as u32;

                view! {
                    <text
                        x=center
                        y=center - 8.0
                        text-anchor="middle"
                        font-size="24"
                        font-weight="bold"
                        fill="white"
                        font-family="Inter, system-ui, sans-serif"
                    >
                        {score}
                    </text>
                    <text
                        x=center
                        y=center + 12.0
                        text-anchor="middle"
                        font-size="10"
                        fill="rgba(255,255,255,0.5)"
                        font-family="Inter, system-ui, sans-serif"
                    >
                        "Overall"
                    </text>
                }
            }}
        </svg>
    }
}
