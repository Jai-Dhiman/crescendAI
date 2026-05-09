"""Content engine CLI entrypoints (typer)."""
from __future__ import annotations
import typer

app = typer.Typer(help="crescendai content engine")


@app.command()
def tick(dry_run: bool = typer.Option(False, "--dry-run", help="No-op for smoke testing")) -> None:
    """Run one orchestrator tick."""
    if dry_run:
        typer.echo("dry-run: orchestrator not invoked")
        return
    typer.echo("tick: orchestrator invoked (real impl wired separately)")


@app.command()
def scout() -> None:
    """Run clip-scout one cycle."""
    typer.echo("scout: not yet implemented in MVP CLI")


@app.command()
def ui() -> None:
    """Start the local Flask UI server."""
    from content_engine.ui.server import build_app
    from content_engine.store.episode_store import EpisodeStore
    import os

    store = EpisodeStore(db_path=os.environ.get("CONTENT_ENGINE_DB", "data/engine.sqlite"))
    flask_app = build_app(episode_store=store)
    flask_app.run(host="127.0.0.1", port=8765, debug=False)


if __name__ == "__main__":
    app()
