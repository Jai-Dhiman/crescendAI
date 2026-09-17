#!/usr/bin/env bash
# MIREX 2026 Track A standardised inference interface.
#
#   predict.sh piece.wav                -> one score on stdout
#   predict.sh a.wav b.wav c.wav        -> one score per line, input order
#   predict.sh /dir/of/wavs             -> one score per line, path order
#   predict.sh --wav-list list.txt      -> one score per line, list order
#   predict.sh --wav-list -             -> read the list from stdin
#   predict.sh -                        -> read the list from stdin
#   predict.sh --format csv <input>     -> '<wav path>,<score>' per line
#
# WHY SO MANY SHAPES. The task page specifies the CONTRACT ("input is a path to
# a WAV file; output is a single real-valued difficulty score") but never the
# INVOCATION: argument vs stdin vs manifest, per-item vs whole-set, stdout vs
# file. The promised reference template has not been published and the captains
# have not answered. Guessing wrong means the container does not run, and a
# container that does not run is excluded from ranking -- so every plausible
# shape is accepted. That asymmetry, not a taste for flexibility, is the reason.
#
# stdout carries ONLY scores. That is not politeness: the MoonBeam fork's
# MusicTokenizer prints its entire vocabulary (~96KB) to stdout on
# construction, so the real output is routed through --out and read back.
# Diagnostics, the SCORE_FAILURE log, and the failure-rate line go to stderr.
#
# BATCH IN ONE INVOCATION WHENEVER POSSIBLE. Every process start reloads the
# 839M backbone; scoring a whole test set one `docker run` at a time would
# spend the 24h budget on model loads rather than on inference.
set -euo pipefail

usage() {
    echo "usage: predict.sh [--format bare|csv] <file.wav|dir|- > [more ...]" >&2
    echo "       predict.sh [--format bare|csv] --wav-list <listfile|- >" >&2
}

# `read` drops a final line that has no trailing newline; the `|| [ -n ... ]`
# clause keeps it. A judge-supplied manifest is not guaranteed to end in \n.
read_into_inputs() {
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        inputs+=("$line")
    done
}

format=bare
inputs=()

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --format)
            [ "$#" -ge 2 ] || { echo "--format takes a value" >&2; exit 2; }
            format="$2"
            shift 2
            ;;
        --wav-list)
            [ "$#" -ge 2 ] || { echo "--wav-list takes exactly one path" >&2; exit 2; }
            if [ "$2" = "-" ]; then
                read_into_inputs
            else
                [ -f "$2" ] || { echo "no such list file: $2" >&2; exit 2; }
                read_into_inputs < "$2"
            fi
            shift 2
            ;;
        -)
            read_into_inputs
            shift
            ;;
        *)
            inputs+=("$1")
            shift
            ;;
    esac
done

case "$format" in
    bare|csv) ;;
    *) echo "--format must be bare or csv, got: $format" >&2; exit 2 ;;
esac

out="$(mktemp)"
listfile="$(mktemp)"
# shellcheck disable=SC2064  # expand the paths now, not at trap time
trap "rm -f '$out' '$listfile'" EXIT

# Normalise every accepted shape into ONE list file, then always invoke
# score_wav.py with --wav-list. Passing a whole test set as repeated --wav
# arguments would risk ARG_MAX, and one code path is one path to test.
for item in ${inputs+"${inputs[@]}"}; do
    item="${item%$'\r'}"   # tolerate a CRLF manifest
    [ -n "$item" ] || continue
    if [ -d "$item" ]; then
        # Recursive and case-insensitive on purpose: a nested or .WAV-suffixed
        # test-set layout must not silently resolve to zero files, which would
        # look like a clean empty run rather than a mistake.
        find "$item" -type f -iname '*.wav' | sort >> "$listfile"
    else
        printf '%s\n' "$item" >> "$listfile"
    fi
done

if [ ! -s "$listfile" ]; then
    echo "no WAV inputs resolved from the arguments given" >&2
    usage
    exit 2
fi

uv run --no-project --script /app/claim_measurement/difficulty/score_wav.py \
    --model-dir "$MIREX_MODEL_DIR" \
    --on-failure "$MIREX_ON_FAILURE" \
    --device "$MIREX_DEVICE" \
    --checkpoint "$MIREX_CHECKPOINT" \
    --repo-root "$MIREX_REPO_ROOT" \
    --model-config "$MIREX_MODEL_CONFIG" \
    --out "$out" \
    --wav-list "$listfile" \
    > /dev/null

# score_wav.py writes '<wav path>\t<score>'; the contract wants the score alone.
if [ "$format" = "csv" ]; then
    tr '\t' ',' < "$out"
else
    cut -f2 "$out"
fi
