# syntax=docker/dockerfile:1
# MIREX 2026 Track A submission container.
#
#   INPUT:  a path to a WAV file (44.1 kHz solo piano)
#   OUTPUT: a single real-valued difficulty score
#
# Pipeline: WAV -> Transkun -> MoonBeam-839M + LoRA -> mean-pool -> ridge -> float
#
# Build the context with build_context.py first -- the model artifacts live
# under the gitignored model/data/ and the 1.6GB checkpoint is not in git, so
# there is no single directory in the repo that can serve as a build context:
#
#     uv run python apps/inference/mirex-difficulty/build_context.py \
#         --model-dir <...>/phase1_lora/model_alldata \
#         --moonbeam-dir <...>/weights/moonbeam \
#         --out-dir /tmp/mirex-ctx
#     docker build -t crescendai/mirex-difficulty /tmp/mirex-ctx
#
# The SUBMISSION build must pass `--platform linux/amd64`: MIREX runs this on an
# x86 CUDA host, and an arm64 image built on an Apple Silicon dev machine will
# not run there. The arm64 build is for local verification only, and only on
# CPU (`-e MIREX_DEVICE=cpu`).
#
# ## Why THREE Python environments, and why that is not accidental complexity
#
# They are mutually incompatible and the repo has already paid to learn it:
#
#   1. The image's own Python -- nothing runs in it; it exists to host uv.
#   2. The MoonBeam env (score_wav.py's PEP-723 `# /// script` header). The
#      fork's vendored transformers hard-requires tokenizers>=0.19,<0.20, and
#      peft is pinned to 0.11.1 -- the version that WROTE these adapters.
#   3. The Transkun env. `transkun_cli.py` deliberately does NOT import
#      transkun: its torch deps conflict with the MoonBeam stack. It shells out
#      to `uv run --no-project --with transkun`, which keeps the two apart.
#
# Merging 2 and 3 is the obvious "simplification" and it is the one thing that
# cannot work. See transkun_cli.py's module docstring.
#
# ## The offline requirement
#
# `uv run --with transkun` would reach the network on first use. A judged run
# cannot depend on that, so BOTH environments are resolved at BUILD time; the
# resulting uv cache is baked into the image and the runtime invocations hit it
# rather than PyPI. UV_OFFLINE=1 at runtime turns a cache miss into a loud
# failure instead of a silent network call that may hang against the 24h budget.

FROM python:3.12-slim AS runtime

# build-essential is required, not defensive: transkun depends on `ncls`, which
# ships no wheel for every platform and falls back to compiling from source.
# Without a compiler the transkun env warm below dies with
# `error: command 'cc' failed: No such file or directory`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 libsndfile1 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV UV_CACHE_DIR=/opt/uv-cache
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# The package layout is load-bearing, not cosmetic: score_wav.py resolves its
# import root as Path(__file__).parents[2] and then imports
# claim_measurement.difficulty.*, so flattening this to /app/score_wav.py would
# resolve the root to / and fail on import.
COPY claim_measurement/ /app/claim_measurement/

# realaudio_check._import_transcribe_wav walks up from CWD/__file__ looking for
# apps/inference/amt/transkun_cli.py. This path is equally load-bearing.
COPY apps/ /app/apps/

# --- environment 3: Transkun, resolved and cached at build time -----------
# The flags must match transkun_cli._run_transkun EXACTLY -- a different spec
# resolves to a different uv cache entry, and UV_OFFLINE would then turn the
# first real transcription into a hard failure.
#
# setuptools<81 is load-bearing twice over: transkun's transcribe.py does
# `import pkg_resources`, and setuptools REMOVED pkg_resources in 81. An
# unpinned spec resolves to a version without it on any cold cache -- which is
# what a Docker build always is. A warm dev cache still holding 80.9.0 hides
# this completely, which is why it surfaced here first.
RUN uv run --no-project --with transkun --with "setuptools<81" --python 3.11 \
        python -c "import transkun, pkg_resources; print('transkun env warm')"

# --- environment 2: MoonBeam, resolved and cached at build time -----------
# --help exercises argparse only; resolving the PEP-723 header is the point --
# it materialises torch, peft==0.11.1 and tokenizers==0.19.1 into the uv cache.
RUN uv run --no-project --script \
        /app/claim_measurement/difficulty/score_wav.py --help > /dev/null \
    && echo "moonbeam env warm"

# --- the model ------------------------------------------------------------
# adapter/ + ridge_head.npz + manifest.json, plus the fork and checkpoint the
# backbone loads against. The 1.6GB checkpoint is baked in rather than fetched:
# a judged run must not depend on the Hub being reachable.
COPY model/    /app/model/
COPY moonbeam/ /app/moonbeam/

ENV MIREX_MODEL_DIR=/app/model
ENV MIREX_CHECKPOINT=/app/moonbeam/moonbeam_839M.pt
ENV MIREX_REPO_ROOT=/app/moonbeam/repo
ENV MIREX_MODEL_CONFIG=/app/moonbeam/repo/src/llama_recipes/configs/model_config.json
# cuda: the contract gives 24 wall-clock hours on ONE GPU for the whole test
# set, which an 839M CPU forward pass per piece does not fit. Override with
# `-e MIREX_DEVICE=cpu` to exercise the container on a machine without a GPU.
ENV MIREX_DEVICE=cuda
# raise while the system is still being built -- a fallback here turns a real
# bug into a plausible-looking median score and we ship the bug. Flip to
# `fallback` for the actual submission, when the contract's >5%-of-items
# exclusion clause dominates. The integration suite covers both.
ENV MIREX_ON_FAILURE=raise
# A cache miss must fail loudly rather than silently reaching PyPI mid-run.
ENV UV_OFFLINE=1

COPY predict.sh /app/predict.sh
RUN chmod +x /app/predict.sh

# The standardised interface: `docker run <image> /path/to/piece.wav` prints one
# real-valued score on stdout and nothing else.
ENTRYPOINT ["/app/predict.sh"]
