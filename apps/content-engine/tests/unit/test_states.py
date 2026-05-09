"""Verifies state transition validity matches the spec's pipeline state machine."""
from content_engine.pipeline.states import State, is_valid_transition


def test_candidate_to_curated_is_valid():
    assert is_valid_transition(State.CANDIDATE, State.CURATED) is True


def test_curated_to_analyzed_is_valid():
    assert is_valid_transition(State.CURATED, State.ANALYZED) is True


def test_candidate_skipping_to_published_is_invalid():
    assert is_valid_transition(State.CANDIDATE, State.PUBLISHED) is False


def test_critic_passed_to_killed_truthfulness_is_invalid():
    assert is_valid_transition(State.CRITIC_PASSED, State.KILLED_TRUTHFULNESS) is False


def test_any_state_to_failure_marker_is_valid():
    assert is_valid_transition(State.ANALYZED, State.FAILED_OBSERVATION) is True


def test_killed_to_critic_passed_via_human_override_is_valid():
    assert is_valid_transition(State.KILLED_TRUTHFULNESS, State.CRITIC_PASSED) is True
