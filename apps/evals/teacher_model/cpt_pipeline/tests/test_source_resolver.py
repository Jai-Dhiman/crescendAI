"""Source resolution behavior tests."""
from teacher_model.cpt_pipeline.source_resolver import resolve_source


def test_youtube_filename_resolves_to_youtube_coarse():
    result = resolve_source("abcdefghijk.txt", {})
    assert result.startswith("youtube:"), f"expected youtube: prefix, got {result!r}"


def test_pdf_filename_resolves_to_academic_pdf_coarse():
    result = resolve_source("pdf_0123456789ab.txt", {})
    assert result.startswith("academic_pdf:"), f"expected academic_pdf: prefix, got {result!r}"
