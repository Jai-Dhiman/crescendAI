"""MIREX 2026 Track A forbidden piece/composer intersection check (#171 S4).

The task call promises a list of forbidden test pieces AND composers. A
*composer* blacklist can retroactively invalidate a trained adapter, because
every adapter was trained on a pool that may include a now-forbidden composer.
This module exists so that publication is a 10-minute run: it answers "does
our training pool intersect the published list, and which fold adapters must
be retrained" against the artifact that was actually trained on.

Authority chain, and why it is three files rather than one:
  * `head_manifest.json` from the staged bundle is the ONLY record of which
    pieces were actually trained on (7,665 seg_ids for the enlarged pool). It
    anonymizes composers to a positional `composer_id`, so it cannot answer
    the composer question alone.
  * `amt_gap_curve/manifest.json` maps seg_id -> PSyllabus key.
  * PSyllabus `new_clean_data.json` maps key -> composer name and title.
The join is verified, not assumed: `check_composer_ids` asserts that grouping
the trained seg_ids by joined composer name reproduces head_manifest's
`composer_id` partition exactly, with `composer_id` equal to the index into
the sorted distinct names. If the join drifts, that assertion fails before any
compliance verdict is printed -- a seg_id whose name failed to join would
otherwise read as "not forbidden", which is the one failure mode this check
exists to prevent.

Matching is deliberately RECALL-FIRST for composers: a false alarm costs a
human five minutes, a miss costs the submission. A forbidden entry matches a
catalog composer when the catalog surname appears as a token in the normalized
forbidden entry ("Chopin", "Chopin, Frederic", "Frederic Chopin" all match the
catalog's "Chopin F."), or when a Wikipedia-derived full-name alias matches
outright. Piece titles use exact normalized equality instead, because
substring matching on titles like "Prelude" matches hundreds of unrelated
pieces; every forbidden entry that matched NOTHING is reported explicitly, so
silence is never mistaken for safety.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def normalize(name: str) -> str:
    """Casefold, strip accents, drop punctuation, collapse whitespace.

    'Chopin F.' -> 'chopin f', 'Frédéric Chopin' -> 'frederic chopin'."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    kept = "".join(c if c.isalnum() or c.isspace() else " " for c in stripped)
    return " ".join(kept.lower().split())


def surname_of(catalog_composer: str) -> str:
    """The catalog's composer strings are 'Surname I.' ('Bach J.S.',
    'Chopin F.'), so the surname is the leading token of the normalized form.
    Multi-token surnames ('Le Couppey F.') keep only the first token, which
    costs recall nothing: the forbidden entry will contain that token too."""
    tokens = normalize(catalog_composer).split()
    return tokens[0] if tokens else ""


def full_name_from_bio_link(link: str | None) -> str | None:
    """PSyllabus records a `composer_bio_link`; the Wikipedia ones carry the
    composer's full name in the slug ('.../wiki/Gabriel_Faur%C3%A9'), which is
    the only place a forbidden list's spelling ('Gabriel Faure') can be met.
    Non-Wikipedia links (pianosyllabus 'artistinfo?name=...') return None."""
    if not link or "wikipedia.org/wiki/" not in link:
        return None
    slug = urllib.parse.unquote(link.rsplit("/", 1)[-1])
    normalized = normalize(slug.replace("_", " "))
    return normalized or None


@dataclass(frozen=True)
class TrainedPiece:
    seg_id: str
    key: str
    composer: str          # catalog string, e.g. "Chopin F."
    title: str             # PSyllabus PS_title
    aliases: frozenset     # normalized alternative spellings of the composer


def build_trained_pool(head_manifest: list, manifest: list, labels: dict) -> list:
    """The trained pool as TrainedPiece rows, one per head_manifest entry.

    Raises on any seg_id that cannot be resolved to a key or to a non-empty
    composer. There is no skip path on purpose: an unresolvable piece is
    exactly the piece that would silently escape the blacklist."""
    key_of = {m["seg_id"]: m["key"] for m in manifest}
    pool = []
    unresolved_key, unresolved_composer = [], []
    for entry in head_manifest:
        seg_id = entry["seg_id"]
        key = key_of.get(seg_id)
        if key is None:
            unresolved_key.append(seg_id)
            continue
        meta = labels.get(key, {})
        composer = str(meta.get("composer", "")).strip()
        if not composer:
            unresolved_composer.append(seg_id)
            continue
        aliases = {normalize(composer)}
        full_name = full_name_from_bio_link(meta.get("composer_bio_link"))
        if full_name:
            aliases.add(full_name)
        pool.append(TrainedPiece(
            seg_id=seg_id, key=key, composer=composer,
            title=str(meta.get("PS_title", "")).strip(),
            aliases=frozenset(aliases)))
    if unresolved_key or unresolved_composer:
        raise ValueError(
            "trained pieces could not be resolved to a composer -- refusing to "
            f"report compliance: {len(unresolved_key)} seg_ids missing from the "
            f"manifest {unresolved_key[:5]}, {len(unresolved_composer)} with no "
            f"composer in the labels {unresolved_composer[:5]}")
    return pool


def check_composer_ids(pool: list, head_manifest: list) -> None:
    """Assert the joined composer names reproduce head_manifest's anonymized
    `composer_id` partition, and that `composer_id` is the index into the
    sorted distinct names (which is how push_train_dataset.build_head_manifest
    assigns it). This is the proof that the join in `build_trained_pool` sees
    the same composers the training run saw."""
    composer_of = {p.seg_id: p.composer for p in pool}
    by_id, by_name = {}, {}
    for entry in head_manifest:
        by_id.setdefault(entry["composer_id"], set()).add(entry["seg_id"])
        by_name.setdefault(composer_of[entry["seg_id"]], set()).add(entry["seg_id"])
    names = sorted(by_name)
    if len(by_id) != len(by_name):
        raise ValueError(
            f"composer join drifted: head_manifest has {len(by_id)} composer "
            f"groups, the labels join produces {len(by_name)}")
    mismatched = [i for i in range(len(names)) if by_id.get(i) != by_name[names[i]]]
    if mismatched:
        raise ValueError(
            "composer join drifted: composer_id groups do not match the sorted "
            f"joined names at ids {mismatched[:5]}")


def match_composers(pool: list, forbidden_composers: list) -> tuple:
    """(matches, unmatched). matches maps each forbidden entry to the sorted
    catalog composers it hits; unmatched lists forbidden entries that hit
    nothing, so a clean run can be distinguished from a mis-parsed list."""
    catalog = {}
    for piece in pool:
        catalog.setdefault(piece.composer, (surname_of(piece.composer), piece.aliases))
    matches, unmatched = {}, []
    for entry in forbidden_composers:
        entry_normalized = normalize(entry)
        tokens = set(entry_normalized.split())
        hits = sorted(
            composer for composer, (surname, aliases) in catalog.items()
            if (surname and surname in tokens) or entry_normalized in aliases)
        if hits:
            matches[entry] = hits
        else:
            unmatched.append(entry)
    return matches, unmatched


def match_pieces(pool: list, forbidden_pieces: list) -> tuple:
    """(matches, unmatched) for piece titles, by exact normalized equality
    against both the PSyllabus title and the catalog key."""
    by_title = {}
    for piece in pool:
        for form in (piece.title, piece.key):
            if form:
                by_title.setdefault(normalize(form), set()).add(piece.seg_id)
    matches, unmatched = {}, []
    for entry in forbidden_pieces:
        hits = sorted(by_title.get(normalize(entry), ()))
        if hits:
            matches[entry] = hits
        else:
            unmatched.append(entry)
    return matches, unmatched


def affected_plans(fold_plans: list, offending_seg_ids: set) -> dict:
    """Which trained plans (the 5 CV folds and the all-data submission plan)
    have a forbidden piece in their train or val split -- i.e. exactly which
    adapters a published list forces us to retrain."""
    affected = {}
    for plan in fold_plans:
        trained_on = (set(plan.get("train_seg_ids", ()))
                      | set(plan.get("val_seg_ids", ())))
        hit = sorted(trained_on & offending_seg_ids)
        if hit:
            affected[plan["fold"]] = hit
    return affected


def run_check(head_manifest: list, manifest: list, labels: dict,
              fold_plans: list, forbidden: dict) -> dict:
    """The whole verdict as a plain dict. `compliant` is True only when no
    forbidden composer and no forbidden piece intersects the trained pool."""
    pool = build_trained_pool(head_manifest, manifest, labels)
    check_composer_ids(pool, head_manifest)

    composer_matches, composer_unmatched = match_composers(
        pool, forbidden.get("composers", []))
    piece_matches, piece_unmatched = match_pieces(pool, forbidden.get("pieces", []))

    hit_composers = {c for hits in composer_matches.values() for c in hits}
    offending = {p.seg_id for p in pool if p.composer in hit_composers}
    offending |= {s for hits in piece_matches.values() for s in hits}

    return {
        "n_trained_pieces": len(pool),
        "n_trained_composers": len({p.composer for p in pool}),
        "composer_matches": composer_matches,
        "composer_unmatched": composer_unmatched,
        "piece_matches": piece_matches,
        "piece_unmatched": piece_unmatched,
        "n_offending_pieces": len(offending),
        "offending_seg_ids": sorted(offending),
        "affected_plans": {f: len(v) for f, v in
                            affected_plans(fold_plans, offending).items()},
        "compliant": not offending,
    }


def format_report(report: dict) -> str:
    lines = [
        f"trained pool: {report['n_trained_pieces']} pieces, "
        f"{report['n_trained_composers']} composers",
    ]
    if report["compliant"]:
        lines.append(
            "COMPLIANT: no forbidden composer or piece is in the trained pool.")
    else:
        lines.append(
            f"NON-COMPLIANT: {report['n_offending_pieces']} trained pieces must be "
            "dropped and the affected adapters retrained.")
        for entry, hits in sorted(report["composer_matches"].items()):
            lines.append(f"  forbidden composer {entry!r} -> catalog {hits}")
        for entry, hits in sorted(report["piece_matches"].items()):
            lines.append(
                f"  forbidden piece {entry!r} -> {len(hits)} seg_ids {hits[:3]}")
        for fold, n in sorted(report["affected_plans"].items()):
            label = "all-data submission plan" if fold == 99 else f"fold {fold}"
            lines.append(f"  retrain {label}: {n} pieces to drop")
    if report["composer_unmatched"]:
        lines.append(
            f"  NOTE: {len(report['composer_unmatched'])} forbidden composers matched "
            f"nothing {report['composer_unmatched'][:5]} -- confirm the list parsed.")
    if report["piece_unmatched"]:
        lines.append(
            f"  NOTE: {len(report['piece_unmatched'])} forbidden pieces matched "
            f"nothing {report['piece_unmatched'][:5]} -- confirm the list parsed.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbidden", type=Path, required=True,
                        help='JSON {"composers": [...], "pieces": [...]} -- the '
                             "published MIREX list, normalized by hand into this "
                             "shape (its published format is not yet known).")
    parser.add_argument("--head-manifest", type=Path, required=True,
                        help="head_manifest.json from the staged training bundle.")
    parser.add_argument("--fold-plans", type=Path, required=True,
                        help="fold_plans.json from the staged training bundle.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = args.manifest or (
        args.data_root / "results" / "amt_gap_curve" / "manifest.json")
    labels_path = args.labels or (
        args.data_root / "raw" / "psyllabus" / "new_clean_data.json")

    report = run_check(
        head_manifest=json.loads(args.head_manifest.read_text()),
        manifest=json.loads(manifest_path.read_text()),
        labels=json.loads(labels_path.read_text()),
        fold_plans=json.loads(args.fold_plans.read_text()),
        forbidden=json.loads(args.forbidden.read_text()))

    print(format_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
    return 0 if report["compliant"] else 1


if __name__ == "__main__":
    sys.exit(main())
