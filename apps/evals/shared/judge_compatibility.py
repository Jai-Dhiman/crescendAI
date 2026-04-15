from __future__ import annotations

# Rules are checked in order; first match wins.
# Each rule is (substring_to_match, family_name).
_FAMILY_RULES: list[tuple[str, str]] = [
    # Anthropic native names
    ("claude-", "anthropic"),
    # OpenRouter slugs (vendor/model)
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("google/", "google"),
    ("qwen/", "qwen"),
    ("meta-llama/", "meta"),
    # Workers AI slugs (@cf/vendor/model)
    ("@cf/openai/", "openai"),
    ("@cf/google/", "google"),
    ("@cf/qwen/", "qwen"),
    ("@cf/meta/", "meta"),
    # Direct vendor names (finetune artifacts, etc.)
    ("gpt-", "openai"),
    ("gemma-", "google"),
    ("gemini-", "google"),
    ("qwen", "qwen"),
    ("llama", "meta"),
]


def model_family(model: str) -> str:
    """Resolve a model name to its family name.

    Raises ValueError if no rule matches (fail-fast on typos).
    """
    lowered = model.lower()
    for substring, family in _FAMILY_RULES:
        if substring in lowered:
            return family
    raise ValueError(f"unknown model family for: {model!r}")


def assert_judge_compatible(teacher_model: str, judge_model: str) -> None:
    """Raise if teacher and judge share a model family.

    Cross-family judging is required to avoid same-family phrasing-preference
    bias in evals. See the eval strategy in
    docs/plans/2026-04-14-eval-improvements.md.
    """
    t_family = model_family(teacher_model)
    j_family = model_family(judge_model)
    if t_family == j_family:
        raise ValueError(
            f"judge family {j_family!r} matches teacher family -- forbidden "
            f"(teacher={teacher_model!r}, judge={judge_model!r})"
        )
