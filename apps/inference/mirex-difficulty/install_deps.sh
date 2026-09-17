#!/usr/bin/env bash
# Host dependencies for the MIREX 2026 Track A difficulty-prediction container.
#
#   ./install_deps.sh          install what is needed to RUN the image
#   ./install_deps.sh --build  also install what is needed to REBUILD it
#
# Target host: Ubuntu 22.04 or 24.04 with an NVIDIA GPU. Everything the model
# itself needs -- Python 3.12, torch, peft, Transkun, the 839M checkpoint -- is
# already inside the image, so the host needs only Docker and the NVIDIA
# container runtime. That is the point of shipping a container.
#
# This script is idempotent and prints what it is about to do. It installs
# system packages and therefore needs sudo.
set -euo pipefail

WANT_BUILD=0
[ "${1:-}" = "--build" ] && WANT_BUILD=1

say() { printf '\n== %s\n' "$*"; }

if [ ! -r /etc/os-release ]; then
    echo "cannot read /etc/os-release; this script targets Ubuntu/Debian" >&2
    exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
case "${ID:-}" in
    ubuntu|debian) ;;
    *) echo "unsupported distro '${ID:-unknown}': this script targets Ubuntu/Debian." >&2
       echo "On another distro, install: docker, nvidia-container-toolkit." >&2
       exit 1 ;;
esac
say "host: ${PRETTY_NAME:-$ID $VERSION_ID}"

# --- 1. NVIDIA driver -------------------------------------------------------
# Not installed here on purpose: the driver is kernel-coupled and usually
# already present and managed on a GPU host. Installing one blindly can break
# a working machine, so this only checks.
say "checking for an NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not found. The container runs on CPU with" >&2
    echo "  -e MIREX_DEVICE=cpu, but the 24h budget assumes a GPU." >&2
fi

# --- 2. Docker --------------------------------------------------------------
say "installing Docker Engine"
if command -v docker >/dev/null 2>&1; then
    echo "docker already present: $(docker --version)"
else
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL "https://download.docker.com/linux/${ID}/gpg" \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                            docker-buildx-plugin
fi

# --- 3. NVIDIA container toolkit -------------------------------------------
# Without this, `docker run --gpus all` fails and the container silently has no
# GPU -- which on this workload means missing the 24h budget, not an error.
say "installing the NVIDIA container toolkit"
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "nvidia-ctk already present: $(nvidia-ctk --version | head -1)"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# --- 4. rebuild-only dependencies ------------------------------------------
if [ "$WANT_BUILD" -eq 1 ]; then
    say "installing rebuild dependencies (uv)"
    if command -v uv >/dev/null 2>&1; then
        echo "uv already present: $(uv --version)"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "uv installed to ~/.local/bin; add it to PATH for this shell:"
        echo '  export PATH="$HOME/.local/bin:$PATH"'
    fi
    echo "NOTE: a rebuild also needs ~30GB free disk and the model artifacts."
    echo "      See README.md, 'Rebuilding from source'."
fi

say "verifying"
docker --version
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "running a GPU smoke test through Docker"
    sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 \
        nvidia-smi --query-gpu=name --format=csv,noheader
fi
say "done. Next: README.md, 'Running the container'."
