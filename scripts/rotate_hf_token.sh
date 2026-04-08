#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="${ROOT_DIR}/ilam_env/.hf_token"

mkdir -p "$(dirname "${TOKEN_FILE}")"

# Remove local project token
rm -f "${TOKEN_FILE}"

# Remove common Hugging Face CLI token caches for current user
rm -f "${HOME}/.huggingface/token"
rm -f "${HOME}/.cache/huggingface/token"
rm -f "${HOME}/.cache/huggingface/stored_tokens"
rm -f "${HOME}/.cache/huggingface/stored_tokens.json"
rm -f "${HOME}/.cache/huggingface/.token"

echo "Enter new Hugging Face token (input hidden):"
read -rs NEW_TOKEN
echo

if [[ -z "${NEW_TOKEN}" ]]; then
  echo "Token is empty. Aborting." >&2
  exit 1
fi

printf "%s\n" "${NEW_TOKEN}" > "${TOKEN_FILE}"
chmod 600 "${TOKEN_FILE}"

echo "Token rotated and saved to ${TOKEN_FILE} with 600 permissions."
