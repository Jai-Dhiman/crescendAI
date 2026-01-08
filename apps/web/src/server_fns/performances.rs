use leptos::prelude::*;

use crate::models::Performance;

#[server(ListPerformances, "/api")]
pub async fn list_performances() -> Result<Vec<Performance>, ServerFnError> {
    Ok(Performance::get_demo_performances())
}

#[server(GetPerformance, "/api")]
pub async fn get_performance(id: String) -> Result<Performance, ServerFnError> {
    Performance::find_by_id(&id)
        .ok_or_else(|| ServerFnError::new("Performance not found"))
}
