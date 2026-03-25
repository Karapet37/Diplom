#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

if [[ -f "${ROOT_DIR}/webapp/package.json" ]]; then
  (
    cd "${ROOT_DIR}/webapp"
    npm install
  )
fi

if [[ ! -f "${ROOT_DIR}/.env.local" && -f "${ROOT_DIR}/.env.example" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env.local"
fi

echo "Bootstrap complete."
echo "Next steps:"
echo "  1. Edit ${ROOT_DIR}/.env.local with your GGUF model paths if autodiscovery is insufficient."
echo "  2. Build the frontend when needed: cd ${ROOT_DIR}/webapp && npm run build"
echo "  3. Run the app: ${ROOT_DIR}/scripts/run_profile.sh development"
