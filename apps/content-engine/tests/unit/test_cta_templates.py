"""Verifies CTA template phase resolution matches spec phases A/B/C."""
import pytest
from content_engine.render.templates import CtaTemplate


def test_phase_a_has_no_in_video_cta():
    tpl = CtaTemplate.for_phase("A")
    assert tpl.end_card_text == ""
    assert tpl.spoken_cta == ""
    assert tpl.watermark_enabled is True


def test_phase_b_has_end_card_and_landing_page():
    tpl = CtaTemplate.for_phase("B")
    assert tpl.end_card_text == "crescend.ai"
    assert tpl.landing_url == "https://crescend.ai/shorts"
    assert tpl.spoken_cta == ""


def test_phase_c_has_spoken_submission_cta():
    tpl = CtaTemplate.for_phase("C")
    assert tpl.spoken_cta != ""
    assert "crescend.ai/submit" in tpl.spoken_cta
    assert tpl.landing_url == "https://crescend.ai/submit"


def test_unknown_phase_raises():
    with pytest.raises(ValueError, match="unknown CTA phase"):
        CtaTemplate.for_phase("Z")
