"""Shared fixtures for cpt_pipeline tests."""
import hashlib
import json
from pathlib import Path

import pytest


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


@pytest.fixture
def tiny_corpus(tmp_path):
    """Hand-crafted 14-file corpus + 3 provenance JSONLs covering every observable behavior.

    Returns (corpus_dir, provenance_dir) as Paths.
    """
    corpus_dir = tmp_path / "corpus"
    provenance_dir = tmp_path / "provenance"
    corpus_dir.mkdir()
    provenance_dir.mkdir()

    # --- Normal docs from three different sources ---
    yt_id_1 = "abcdefghijk"
    (corpus_dir / f"{yt_id_1}.txt").write_text(
        "Practicing scales builds finger independence.\n\n"
        "Start slow with a metronome at 60 bpm. Increase by 4 bpm only after\n"
        "you can play four consecutive perfect repetitions. The Russian school\n"
        "emphasizes weight transfer through the keys, while the Taubman approach\n"
        "focuses on rotation and alignment of the forearm.\n",
        encoding="utf-8",
    )

    yt_id_2 = "lmnopqrstuv"
    (corpus_dir / f"{yt_id_2}.txt").write_text(
        "Voicing the melody in a Chopin nocturne requires careful balance.\n\n"
        "The right hand top voice should sing above the inner accompaniment\n"
        "voices. Practice each voice separately, then layer them with awareness\n"
        "of dynamic hierarchy.\n",
        encoding="utf-8",
    )

    pdf_url_1 = "https://openalex.example.org/W123/paper.pdf"
    pdf_h_1 = _sha12(pdf_url_1)
    (corpus_dir / f"pdf_{pdf_h_1}.txt").write_text(
        "Pedagogical approaches to early-stage technique acquisition.\n\n"
        "This paper examines how teachers structure the first six months of\n"
        "instruction, with particular attention to posture, hand position,\n"
        "and tone production. The authors interviewed forty conservatory\n"
        "professors across three continents.\n\n"
        "References\n"
        "Smith, J. (2019). Foundations of piano pedagogy. Journal of Piano X.\n"
        "Lee, A. (2020). Tone production studies. Pedagogy Quarterly.\n",
        encoding="utf-8",
    )

    # --- Doc with legal disclaimer boilerplate (corpus-wide line-freq target) ---
    web_url_disc = "https://music.example.org/disclaimer/article-1"
    web_h_disc = _sha12(web_url_disc)
    boilerplate_line = (
        "The author and publisher disclaim all such representations and warranties for a particular purpose."
    )
    (corpus_dir / f"web_{web_h_disc}.txt").write_text(
        f"Real content about phrasing.\n\n"
        f"{boilerplate_line}\n"
        f"Phrasing means shaping a musical line so its peaks and resolutions\n"
        f"feel inevitable to the listener.\n",
        encoding="utf-8",
    )

    # 24 more web docs containing the same boilerplate line — pushes >20 threshold
    for i in range(24):
        url = f"https://music.example.org/disclaimer/article-{i + 2}"
        h = _sha12(url)
        (corpus_dir / f"web_{h}.txt").write_text(
            f"Doc {i} body content here. Talking about ornamentation in baroque music\n"
            f"requires understanding the conventions of the period.\n\n"
            f"{boilerplate_line}\n"
            f"More body content about appoggiaturas and trills.\n",
            encoding="utf-8",
        )

    # --- Doc with within-doc repeated lines (within-doc strip target) ---
    yt_id_repeats = "wxyz12345AB"
    repeat_block = "Newly formed bands\nAlbums\nDisbandments\nEvents\n"
    (corpus_dir / f"{yt_id_repeats}.txt").write_text(
        "Year-by-year history of music ensembles.\n\n"
        f"1972\n{repeat_block}1973\n{repeat_block}1974\n{repeat_block}"
        f"1975\n{repeat_block}1976\n{repeat_block}\n"
        "End of the historical survey section.\n",
        encoding="utf-8",
    )

    # --- Two academic-paper docs with References section ---
    pdf_url_refs = "https://openalex.example.org/W456/paper.pdf"
    pdf_h_refs = _sha12(pdf_url_refs)
    (corpus_dir / f"pdf_{pdf_h_refs}.txt").write_text(
        "Body content about pedaling techniques in Debussy.\n\n"
        "The half-pedal allows partial damper engagement, useful for impressionist textures.\n\n"
        "References\n"
        "Debussy, C. (1905). Estampes. Durand.\n"
        "Howat, R. (1983). Debussy in proportion. Cambridge University Press.\n",
        encoding="utf-8",
    )

    # --- Two near-duplicate docs (Jaccard ~0.97 at 5-char shingles, above 0.93 threshold) ---
    yt_id_dup_a = "DUP1234567x"
    yt_id_dup_b = "DUP1234567y"
    dup_text_base = (
        "Slow practice is the foundation of all technique work. Begin every\n"
        "session with five minutes of metronome scales at quarter = 60. The\n"
        "objective is not speed but evenness, articulation, and tonal control.\n"
        "Listen for unevenness in the weaker fingers and isolate problem groups.\n"
    )
    (corpus_dir / f"{yt_id_dup_a}.txt").write_text(dup_text_base, encoding="utf-8")
    (corpus_dir / f"{yt_id_dup_b}.txt").write_text(
        dup_text_base + "Extra.\n", encoding="utf-8"
    )

    # --- Doc <100 chars (length-floor drop) ---
    yt_id_short = "SHORTABCDEF"
    (corpus_dir / f"{yt_id_short}.txt").write_text("Too short.", encoding="utf-8")

    # --- Doc with 60% non-ASCII ---
    yt_id_nonascii = "NONASCIIABC"
    (corpus_dir / f"{yt_id_nonascii}.txt").write_text(
        "Mostly content here. " + ("中文内容" * 30),
        encoding="utf-8",
    )

    # --- Doc in French (language-filter drop) ---
    yt_id_fr = "FRENCHABCDE"
    (corpus_dir / f"{yt_id_fr}.txt").write_text(
        "La pratique du piano est un art exigeant qui demande de la patience.\n"
        "Les exercices de Hanon developpent la force et l'independance des doigts.\n"
        "Chaque journee de pratique doit commencer par un echauffement progressif.\n",
        encoding="utf-8",
    )

    # --- Corrupt UTF-8 file ---
    yt_id_corrupt = "CORRUPTABCD"
    (corpus_dir / f"{yt_id_corrupt}.txt").write_bytes(
        b"Some valid prefix text.\n\xff\xfe\xfd\xfc invalid UTF-8 here \xc3\x28 broken.\n"
    )

    # --- Provenance JSONLs ---
    yt_jsonl = provenance_dir / "provenance_tonebase.jsonl"
    yt_jsonl.write_text(
        json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_1}",
            "title": "Scale Practice Masterclass",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-01T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 50,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_2}",
            "title": "Chopin Voicing",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-02T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 40,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_repeats}",
            "title": "Music History Survey",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-03T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_dup_a}",
            "title": "Slow Practice 1",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-04T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 50,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_dup_b}",
            "title": "Slow Practice 2",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-05T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 55,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_fr}",
            "title": "Pratique Francaise",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-06T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier1_youtube",
        }) + "\n",
        encoding="utf-8",
    )

    pdf_jsonl = provenance_dir / "provenance_openalex.jsonl"
    pdf_jsonl.write_text(
        json.dumps({
            "url": pdf_url_1,
            "title": "Pedagogy Paper 1",
            "channel_or_publisher": "OpenAlex",
            "download_timestamp": "2026-04-07T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 60,
            "source_tier": "tier3_musicology",
        }) + "\n" + json.dumps({
            "url": pdf_url_refs,
            "title": "Debussy Pedaling",
            "channel_or_publisher": "OpenAlex",
            "download_timestamp": "2026-04-08T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 80,
            "source_tier": "tier3_musicology",
        }) + "\n",
        encoding="utf-8",
    )

    web_jsonl = provenance_dir / "provenance_disclaimer.jsonl"
    web_jsonl.write_text(
        json.dumps({
            "url": web_url_disc,
            "title": "Phrasing article",
            "channel_or_publisher": "music.example.org",
            "download_timestamp": "2026-04-09T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier2_literature",
        }) + "\n",
        encoding="utf-8",
    )

    return corpus_dir, provenance_dir


@pytest.fixture
def fixture_ids(tiny_corpus):
    """Stable handle on the doc_ids the fixture produces, for assertions."""
    corpus_dir, _ = tiny_corpus
    return {
        "yt_normal_1": "abcdefghijk",
        "yt_normal_2": "lmnopqrstuv",
        "yt_repeats": "wxyz12345AB",
        "yt_short": "SHORTABCDEF",
        "yt_nonascii": "NONASCIIABC",
        "yt_french": "FRENCHABCDE",
        "yt_corrupt": "CORRUPTABCD",
        "yt_dup_a": "DUP1234567x",
        "yt_dup_b": "DUP1234567y",
        "pdf_h_1": _sha12("https://openalex.example.org/W123/paper.pdf"),
        "pdf_h_refs": _sha12("https://openalex.example.org/W456/paper.pdf"),
        "web_h_disc": _sha12("https://music.example.org/disclaimer/article-1"),
    }
