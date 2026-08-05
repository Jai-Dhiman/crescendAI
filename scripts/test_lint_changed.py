"""Tests for the changed-lines lint filter (#147)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lint_changed import Finding, parse_changed_lines, partition


def _f(path: str, line: int | None, code: str = "lint/x") -> Finding:
    return Finding("biome", path, line, code, "error", "msg")


class TestParseChangedLines:
    def test_reads_new_file_range_from_hunk_header(self):
        diff = "@@ -10,3 +12,4 @@ def thing():\n+added\n"

        assert parse_changed_lines(diff) == {12, 13, 14, 15}

    def test_absent_count_means_one_line(self):
        diff = "@@ -5 +7 @@\n"

        assert parse_changed_lines(diff) == {7}

    def test_pure_deletion_contributes_nothing(self):
        # "+9,0" means lines were removed and none added at 9.
        diff = "@@ -9,2 +9,0 @@\n"

        assert parse_changed_lines(diff) == set()

    def test_multiple_hunks_union(self):
        diff = "@@ -1,1 +1,2 @@\n@@ -50,0 +60,3 @@\n"

        assert parse_changed_lines(diff) == {1, 2, 60, 61, 62}

    def test_added_file_covers_every_line(self):
        diff = "@@ -0,0 +1,5 @@\n"

        assert parse_changed_lines(diff) == {1, 2, 3, 4, 5}

    def test_ignores_content_lines_that_look_like_headers(self):
        # A removed line of source that happens to start with @@ must not be
        # mistaken for a hunk header -- real headers are not prefixed.
        diff = "@@ -1,1 +1,1 @@\n-@@ -999,5 +999,5 @@ not a header\n"

        assert parse_changed_lines(diff) == {1}

    def test_empty_diff(self):
        assert parse_changed_lines("") == set()


class TestPartition:
    def test_finding_on_a_changed_line_blocks(self):
        blocking, advisory = partition([_f("a.ts", 12)], {"a.ts": {12}})

        assert len(blocking) == 1
        assert advisory == []

    def test_finding_on_an_untouched_line_is_advisory(self):
        # The #143 case: 19 findings at lines 1179-2081 while the commit
        # touched ~110-175. None should block.
        blocking, advisory = partition([_f("a.ts", 1179)], {"a.ts": {110, 111}})

        assert blocking == []
        assert len(advisory) == 1

    def test_whole_file_finding_never_blocks(self):
        blocking, advisory = partition([_f("a.ts", None)], {"a.ts": {1, 2, 3}})

        assert blocking == []
        assert len(advisory) == 1

    def test_lines_are_matched_per_file_not_globally(self):
        # Line 12 is changed in a.ts but not b.ts; only a.ts's finding blocks.
        findings = [_f("a.ts", 12), _f("b.ts", 12)]
        blocking, advisory = partition(findings, {"a.ts": {12}, "b.ts": {99}})

        assert [f.path for f in blocking] == ["a.ts"]
        assert [f.path for f in advisory] == ["b.ts"]

    def test_file_with_no_changed_line_entry_is_advisory(self):
        blocking, advisory = partition([_f("c.ts", 5)], {})

        assert blocking == []
        assert len(advisory) == 1

    def test_mixed_set_splits_both_ways(self):
        findings = [_f("a.ts", 10), _f("a.ts", 500), _f("a.ts", None)]
        blocking, advisory = partition(findings, {"a.ts": {10, 11}})

        assert [f.line for f in blocking] == [10]
        assert sorted(str(f.line) for f in advisory) == ["500", "None"]
