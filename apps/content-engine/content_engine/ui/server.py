"""Local Flask server: swipe-review + voiceover-record + final-approve UIs."""
from __future__ import annotations
from flask import Flask, jsonify, render_template
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore, InvalidTransitionError


def build_app(episode_store: EpisodeStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def root() -> str:
        return render_template("swipe.html")

    @app.post("/swipe/<episode_id>/approve")
    def approve(episode_id: str):
        try:
            episode_store.transition(episode_id, State.CURATED)
        except KeyError:
            return jsonify({"ok": False, "error": "episode not found"}), 404
        except InvalidTransitionError:
            return jsonify({"ok": False, "error": "invalid transition"}), 409
        return jsonify({"ok": True, "state": "curated"})

    @app.post("/swipe/<episode_id>/reject")
    def reject(episode_id: str):
        return jsonify({"ok": True, "state": "candidate"})

    return app
