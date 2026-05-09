from teacher_model.calibration.era_lookup import composer_to_era


def test_known_composers_map_to_expected_eras():
    assert composer_to_era("Bach") == "Baroque"
    assert composer_to_era("Beethoven") == "Classical"
    assert composer_to_era("Chopin") == "Romantic"
    assert composer_to_era("Debussy") == "Impressionist"


def test_unknown_composer_returns_other():
    assert composer_to_era("Stravinsky") == "Other"
    assert composer_to_era("") == "Other"


def test_known_composers_are_case_sensitive_match():
    # Spec choice: exact-case match only, since baseline_v1.jsonl uses canonical
    # capitalization. Lowercased input is treated as unknown rather than silently
    # normalized — surfaces data quality issues instead of hiding them.
    assert composer_to_era("bach") == "Other"
