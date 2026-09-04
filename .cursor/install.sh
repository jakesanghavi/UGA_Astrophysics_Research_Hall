#!/usr/bin/env bash
# Idempotent environment bootstrap for the UGA Astrophysics Research Hall repo.
# Installs the system toolchain needed to compile the ExoPlaSim (PlaSim) Fortran
# climate model, then creates a Python virtual environment with all project deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- System toolchain (compilers + MPI) required by exoplasim ---
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc g++ gfortran \
    libopenmpi-dev openmpi-bin \
    python3-venv python3-dev

# --- Project Python virtual environment ---
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Pre-compile ExoPlaSim's pyfft postprocessor library so the first model run
# does not pay the one-time compilation cost.
python -c "import exoplasim; exoplasim.sysconfigure()"

echo "Environment ready. Activate with: source .venv/bin/activate"
