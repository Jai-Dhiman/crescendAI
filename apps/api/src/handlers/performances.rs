use crate::models::Performance;
use crate::utils::{error_response, json_response};
use worker::{Request, Response, Result, RouteContext};

/// GET /api/performances
/// Returns the list of all demo performances.
pub async fn handle_list_performances<D>(
    _req: Request,
    _ctx: RouteContext<D>,
) -> Result<Response> {
    let performances = Performance::get_demo_performances();
    json_response(&performances)
}

/// GET /api/performances/:id
/// Returns a single performance by ID.
pub async fn handle_get_performance<D>(
    _req: Request,
    ctx: RouteContext<D>,
) -> Result<Response> {
    let id = match ctx.param("id") {
        Some(id) => id,
        None => return error_response("Missing performance ID", 400),
    };

    match Performance::find_by_id(id) {
        Some(performance) => json_response(&performance),
        None => error_response("Performance not found", 404),
    }
}
