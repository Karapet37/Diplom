#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-development}"

if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="python3"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

ENV_FILE=""
if [[ -n "${COGNITIVE_ENV_FILE:-}" ]]; then
  ENV_FILE="${COGNITIVE_ENV_FILE}"
elif [[ -f "${ROOT_DIR}/.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/.env.local"
elif [[ -f "${ROOT_DIR}/.env" ]]; then
  ENV_FILE="${ROOT_DIR}/.env"
fi

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/start.py" "--profile" "${PROFILE}")
if [[ -n "${ENV_FILE}" ]]; then
  CMD+=("--env-file" "${ENV_FILE}")
fi
CMD+=("$@")

echo "Launching Persona-Graph Agent System"
echo "  profile: ${PROFILE}"
if [[ -n "${ENV_FILE}" ]]; then
  echo "  env: ${ENV_FILE}"
fi
exec "${CMD[@]}"
