"""Pipeline state machine: states and valid transitions."""
from enum import Enum


class State(str, Enum):
    CANDIDATE = "candidate"
    CURATED = "curated"
    ANALYZED = "analyzed"
    OBSERVATION_SELECTED = "observation_selected"
    SCRIPT_DRAFTED = "script_drafted"
    KILLED_TRUTHFULNESS = "killed_truthfulness"
    CRITIC_PASSED = "critic_passed"
    RECORDED = "recorded"
    RENDERED = "rendered"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    MEASURED = "measured"
    FAILED_ANALYSIS = "failed_analysis"
    FAILED_OBSERVATION = "failed_observation"
    FAILED_SCRIPT = "failed_script"
    FAILED_CRITIC = "failed_critic"
    FAILED_RENDER = "failed_render"
    FAILED_SCHEDULE = "failed_schedule"


_FORWARD: dict[State, set[State]] = {
    State.CANDIDATE: {State.CURATED},
    State.CURATED: {State.ANALYZED, State.FAILED_ANALYSIS},
    State.ANALYZED: {State.OBSERVATION_SELECTED, State.FAILED_OBSERVATION},
    State.OBSERVATION_SELECTED: {State.SCRIPT_DRAFTED, State.FAILED_SCRIPT},
    State.SCRIPT_DRAFTED: {State.CRITIC_PASSED, State.KILLED_TRUTHFULNESS, State.FAILED_CRITIC},
    State.KILLED_TRUTHFULNESS: {State.CRITIC_PASSED},
    State.CRITIC_PASSED: {State.RECORDED},
    State.RECORDED: {State.RENDERED, State.FAILED_RENDER},
    State.RENDERED: {State.SCHEDULED, State.FAILED_SCHEDULE},
    State.SCHEDULED: {State.PUBLISHED, State.FAILED_SCHEDULE},
    State.PUBLISHED: {State.MEASURED},
    State.MEASURED: set(),
    State.FAILED_ANALYSIS: {State.ANALYZED},
    State.FAILED_OBSERVATION: {State.OBSERVATION_SELECTED},
    State.FAILED_SCRIPT: {State.SCRIPT_DRAFTED},
    State.FAILED_CRITIC: {State.CRITIC_PASSED, State.KILLED_TRUTHFULNESS},
    State.FAILED_RENDER: {State.RENDERED},
    State.FAILED_SCHEDULE: {State.SCHEDULED, State.PUBLISHED},
}


def is_valid_transition(src: State, dst: State) -> bool:
    return dst in _FORWARD.get(src, set())
