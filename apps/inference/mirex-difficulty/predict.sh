#!/usr/bin/env bash
# MIREX 2026 Track A standardised inference interface.
#
#   predict.sh /path/to/piece.wav          -> one score on stdout
#   predict.sh a.wav b.wav c.wav           -> one score per line, input order
#   predict.sh --wav-list /path/to/list    -> one score per line, list order
#
# stdout carries ONLY scores. That is not politeness: the MoonBeam fork's
# MusicTokenizer prints its entire vocabulary (~96KB) to stdout on
# construction, so the real output is routed through --out and read back.
# Diagnostics, the SCORE_FAILURE log, and the failure-rate line go to stderr.
#
# BATCH IN ONE INVOCATION WHENEVER POSSIBLE. Every process start reloads the
# 839M backbone; scoring a whole test set one `docker run` at a time would
# spend the 24h budget on model loads rather than on inference. How MIREX
# actually invokes this is under-specified -- an open question to the captains
# -- so both shapes are supported.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: predict.sh <file.wav> [more.wav ...] | --wav-list <listfile>" >&2
    exit 2
fi

out="$(mktemp)"
# shellcheck disable=SC2064  # expand $out now, not at trap time
trap "rm -f '$out'" EXIT

args=()
if [ "$1" = "--wav-list" ]; then
    if [ "$#" -ne 2 ]; then
        echo "--wav-list takes exactly one path" >&2
        exit 2
    fi
    args=(--wav-list "$2")
else
    for wav in "$@"; do
        args+=(--wav "$wav")
    done
fi

uv run --no-project --script /app/claim_measurement/difficulty/score_wav.py \
    --model-dir "$MIREX_MODEL_DIR" \
    --on-failure "$MIREX_ON_FAILURE" \
    --device "$MIREX_DEVICE" \
    --checkpoint "$MIREX_CHECKPOINT" \
    --repo-root "$MIREX_REPO_ROOT" \
    --model-config "$MIREX_MODEL_CONFIG" \
    --out "$out" \
    "${args[@]}" \
    > /dev/null

# score_wav.py writes '<wav_path>\t<score>'; the contract wants the score alone.
cut -f2 "$out"
