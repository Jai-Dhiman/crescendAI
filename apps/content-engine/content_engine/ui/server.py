"""Local Flask server: swipe-review + voiceover-record + final-approve UIs."""
from __future__ import annotations
from flask import Flask, jsonify, render_template
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore


def build_app(episode_store: EpisodeStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def root() -> str:
        return render_template("swipe.html")

    @app.post("/swipe/<episode_id>/approve")
    def approve(episode_id: str):
        episode_store.transition(episode_id, State.CURATED)
        return jsonify({"ok": True, "state": "curated"})

    @app.post("/swipe/<episode_id>/reject")
    def reject(episode_id: str):
        return jsonify({"ok": True, "state": "candidate"})

    return app
