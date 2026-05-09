"""Local Flask server: swipe-review + voiceover-record + final-approve UIs."""
from __future__ import annotations
from flask import Flask, jsonify, render_template, request
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
        if episode_store.get(episode_id) is None:
            return jsonify({"ok": False, "error": "episode not found"}), 404
        return jsonify({"ok": True, "state": "candidate"})

    @app.post("/swipe/<episode_id>/override-critic")
    def override_critic(episode_id: str):
        try:
            episode_store.transition(episode_id, State.CRITIC_PASSED)
        except KeyError:
            return jsonify({"ok": False, "error": "episode not found"}), 404
        except InvalidTransitionError:
            return jsonify({"ok": False, "error": "invalid transition"}), 409
        return jsonify({"ok": True, "state": "critic_passed"})

    @app.post("/record/<episode_id>/complete")
    def record_complete(episode_id: str):
        body = request.get_json(silent=True) or {}
        voiceover_path = body.get("voiceover_path")
        if not voiceover_path:
            return jsonify({"ok": False, "error": "voiceover_path required"}), 400
        try:
            ep = episode_store.get(episode_id)
            if ep is None:
                return jsonify({"ok": False, "error": "episode not found"}), 404
            ep.voiceover_path = voiceover_path
            episode_store.save(ep)
            episode_store.transition(episode_id, State.RECORDED)
        except InvalidTransitionError:
            return jsonify({"ok": False, "error": "invalid transition"}), 409
        return jsonify({"ok": True, "state": "recorded"})

    return app
