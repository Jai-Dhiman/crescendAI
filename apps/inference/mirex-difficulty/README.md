# CrescendAI — MIREX 2026 Music Performance Difficulty Prediction

A Docker container that takes a path to a solo-piano WAV file and prints a
single real-valued difficulty score.

Pipeline: **WAV → Transkun transcription → MoonBeam-839M with LoRA adapters →
mean-pool → ridge head → float.** The submitted system is a 5-member ensemble
whose members differ only by training seed; their predictions are averaged, so
the container emits one score.

---

## 1. Host requirements

| | |
|---|---|
| Host OS | Ubuntu 22.04 or 24.04 (any Linux with Docker and an NVIDIA GPU works) |
| Image OS | Debian 13 (trixie), from `python:3.12.14-slim-trixie` |
| GPU | One NVIDIA GPU. CPU works but will not meet the 24h budget |
| Host software | Docker Engine with buildx, NVIDIA container toolkit |
| Disk | ~21GB for the image |
| Network | **None at run time.** The image is fully self-contained |

Everything the model needs — Python 3.12, torch, `peft==0.11.1`, Transkun, the
1.6GB MoonBeam checkpoint — is inside the image. The host needs only Docker and
the NVIDIA runtime.

The image base is pinned to `python:3.12.14-slim-trixie` rather than
`python:3.12-slim`. That tag is a floating alias which has already moved from
Debian 12 to Debian 13; an unpinned base would make the OS declared above
untrue after the next rebuild.

### Installing the host dependencies

```bash
./install_deps.sh            # to RUN the image
./install_deps.sh --build    # also install uv, to REBUILD it
```

The script installs Docker Engine and the NVIDIA container toolkit, and
verifies both. It deliberately does **not** install an NVIDIA driver: the
driver is kernel-coupled and normally already managed on a GPU host, so the
script checks for one and warns rather than risking a working machine.

---

## 2. Running the container

Load the image, then invoke it. Every form below is one process, and
**batching matters**: each start reloads the 839M backbone, so scoring a whole
test set one `docker run` at a time spends the budget on model loads.

```bash
docker load -i crescendai-mirex-difficulty.tar

RUN="docker run --rm --gpus all -v /path/to/audio:/audio:ro crescendai/mirex-difficulty"

$RUN /audio/piece.wav                    # one score on stdout
$RUN /audio/a.wav /audio/b.wav           # one score per line, input order
$RUN /audio                              # every .wav below the directory, sorted
$RUN --wav-list /audio/list.txt          # one score per line, list order
cat list.txt | $RUN --wav-list -         # the list on stdin
$RUN --format csv /audio                 # '<path>,<score>' per line
```

### Interface contract

- **stdout carries only scores**, one per line, in input order. Nothing else
  is ever written there. (This is enforced, not incidental: the MoonBeam
  tokenizer prints ~96KB of vocabulary on construction, so real output is
  routed through a file and read back.)
- **stderr carries everything else** — progress, `SCORE_FAILURE` lines, the
  failure-rate summary.
- **Exit 0** = a score was emitted for every requested item. Note that under
  the default `fallback` policy this includes items that fell back to the
  median, so exit 0 is not by itself evidence that every item transcribed
  cleanly — check stderr for `SCORE_FAILURE`. **Exit 2** = a usage error,
  raised before any model is loaded. Non-zero otherwise.
- `--format csv` pairs each score with its path explicitly, for when
  positional pairing is not convenient.

Multiple input shapes are supported because the task page specifies the
contract (WAV path in, one float out) but not the calling convention, and the
promised reference template has not been published. Use whichever fits your
harness.

### Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `MIREX_DEVICE` | `cuda` | Set `cpu` to run without a GPU (much slower) |
| `MIREX_ON_FAILURE` | `fallback` | `raise` to make a failing item abort, for debugging |

**On `fallback`.** The task rules exclude any submission failing on more than
5% of items, so a single unhandled exception is far more costly than a
defensible score. On failure the container emits the corpus median and logs
`SCORE_FAILURE <path>: <error>` to stderr. It is loud but not fatal — if
anything went wrong, it is in your log.

---

## 3. Compute

Measured on one NVIDIA L4 (`l4x1`), 8 real recordings spanning 15s–1806s,
0/8 failures:

| | per item |
|---|---|
| 5-member ensemble (submitted) | **4.97s + 0.0400 × duration** |
| single adapter (reference) | 4.93s + 0.0360 × duration |

At our corpus's median piece length of 103s that is **≈9,500 items per 24h on
one GPU**. Encoder cost is linear in duration, not a per-piece constant, so
scale the slope rather than the fixed term when estimating a different test
set.

Peak GPU memory was not measured and is not declared.

---

## 4. Rebuilding from source

Needs `--build` dependencies, ~30GB free disk, and the model artifacts.

```bash
# 1. Fetch the five ensemble members (adapter + training embeddings each).
for s in 2026 2027 2028 2029 2030; do
    hf download "Jai-D/phase1-lora-7665-alldata-s${s}" \
        --local-dir "artifacts/alldata-s${s}"
done

# 2. Assemble the five-member model directory.
cd model && uv run python -m claim_measurement.difficulty.build_model_dir \
    --from-ensemble ../artifacts/alldata-s2026 \
    --from-ensemble ../artifacts/alldata-s2027 \
    --from-ensemble ../artifacts/alldata-s2028 \
    --from-ensemble ../artifacts/alldata-s2029 \
    --from-ensemble ../artifacts/alldata-s2030 \
    --out-dir data/results/phase1_lora/model_ensemble5
cd ..

# 3. Assemble the Docker build context (~5GB; written outside the repo).
uv run python apps/inference/mirex-difficulty/build_context.py \
    --model-dir  model/data/results/phase1_lora/model_ensemble5 \
    --moonbeam-dir model/data/weights/moonbeam \
    --out-dir /tmp/mirex-ctx

# 4. Build. --platform is REQUIRED for the submission image: MIREX runs this on
#    an x86 CUDA host, and an arm64 image built on Apple Silicon will not run.
docker build --platform linux/amd64 \
    -t crescendai/mirex-difficulty /tmp/mirex-ctx

# 5. Export for submission.
docker save crescendai/mirex-difficulty -o crescendai-mirex-difficulty.tar
```

### Why the image holds three Python environments

They are mutually incompatible, and the repository has already paid to learn
it. The image's own Python only hosts `uv`. The MoonBeam environment pins
`tokenizers==0.19.1` (the fork's vendored transformers requires `>=0.19,<0.20`)
and `peft==0.11.1` (the version that wrote these adapters). Transkun's torch
dependencies conflict with both, so it is shelled out to a separate `uv`
environment. Merging them is the obvious simplification and the one thing that
cannot work.

Both environments are resolved at **build** time and their `uv` cache is baked
into the image; `UV_OFFLINE=1` at run time turns a cache miss into a loud
failure rather than a silent network call that could hang against the budget.

---

## 5. Verifying a build

```bash
# Interface tests -- no image needed, seconds.
cd model && uv run --project . python -m pytest \
    ../apps/inference/mirex-difficulty/test_predict_sh.py \
    ../apps/inference/mirex-difficulty/test_build_context.py -q --no-cov

# Container tests -- needs the built image and some WAVs.
CRESCENDAI_RUN_CONTAINER_TESTS=1 \
CRESCENDAI_WAV_DIR=/path/to/wavs \
uv run python -m pytest ../apps/inference/mirex-difficulty/test_container.py \
    -q --no-cov -rs
```

**Check that scores vary.** Because the shipped default is
`MIREX_ON_FAILURE=fallback`, a completely broken model directory would produce
a clean run of identical median scores and pass every exit-code check. Scoring
two different pieces must give two different numbers.

---

## 6. Contact

Jai Dhiman — jaidhiman2000@gmail.com
