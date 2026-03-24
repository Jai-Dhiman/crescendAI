"""Structural tests for eval infrastructure."""


def test_multi_layer_parsing():
    """Multiple --layer flags should all be recognized."""
    from src.run_all import parse_args
    args = parse_args(["--layer", "synthesis", "--layer", "temporal"])
    assert "synthesis" in args.layers
    assert "temporal" in args.layers
    assert "retrieval" not in args.layers


def test_all_layers_default():
    """No --layer flag should run all layers."""
    from src.run_all import parse_args
    args = parse_args([])
    assert len(args.layers) == len(["retrieval", "synthesis", "temporal", "downstream", "chat_extraction", "locomo", "report"])


def test_json_output_flag():
    """--json-output should be recognized."""
    from src.run_all import parse_args
    args = parse_args(["--json-output"])
    assert args.json_output is True


def test_composite_formula():
    """Composite formula should be 0.4*synth + 0.3*temp + 0.3*chat."""
    from src.run_all import compute_composite
    assert abs(compute_composite(1.0, 1.0, 1.0) - 1.0) < 0.001
    assert abs(compute_composite(0.5, 0.5, 0.5) - 0.5) < 0.001
    expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.7
    assert abs(compute_composite(0.8, 0.6, 0.7) - expected) < 0.001


def test_composite_reweight_without_chat():
    """When chat_extraction unavailable, reweight to 0.55/0.45."""
    from src.run_all import compute_composite_without_chat
    expected = 0.55 * 0.8 + 0.45 * 0.6
    assert abs(compute_composite_without_chat(0.8, 0.6) - expected) < 0.001


def test_realistic_scenario_schema():
    """Generated scenarios must have required fields."""
    from src.scenarios import MemoryEvalScenario, Observation
    from dataclasses import fields
    required_obs_fields = {"id", "dimension", "observation_text", "session_id", "session_date"}
    actual_obs_fields = {f.name for f in fields(Observation)}
    assert required_obs_fields.issubset(actual_obs_fields)
    required_scenario_fields = {"id", "name", "category", "observations", "checkpoints", "expected_facts", "version"}
    actual_scenario_fields = {f.name for f in fields(MemoryEvalScenario)}
    assert required_scenario_fields.issubset(actual_scenario_fields)
