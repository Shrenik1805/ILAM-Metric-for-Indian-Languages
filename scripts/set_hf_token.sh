#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="${ROOT_DIR}/ilam_env/.hf_token"

mkdir -p "$(dirname "${TOKEN_FILE}")"

if [[ "${1:-}" != "" ]]; then
  TOKEN_VALUE="$1"
else
  echo "Enter Hugging Face token (input hidden):"
  read -rs TOKEN_VALUE
  echo
fi

if [[ -z "${TOKEN_VALUE}" ]]; then
  echo "Token is empty. Aborting." >&2
  exit 1
fi

printf "%s\n" "${TOKEN_VALUE}" > "${TOKEN_FILE}"
chmod 600 "${TOKEN_FILE}"

echo "Token saved to ${TOKEN_FILE} with 600 permissions."
