"""
ilam.hf_auth
------------
Shared Hugging Face token loader for this project/package.

Priority:
1. Environment variables:
   ILAM_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_TOKEN
2. Explicit token file path via env var: ILAM_HF_TOKEN_FILE
3. Project-local token file: <repo>/ilam_env/.hf_token (if present)
4. CWD-local token file: ./ilam_env/.hf_token (if present)

Notes:
- For a distributable package, environment variables (or `huggingface-cli login`)
  are the recommended approach. The project-local `ilam_env/.hf_token` is
  supported for this repo's workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_KEYS = (
    "ILAM_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)

_WARNED_INSECURE_PERMS = False


def _candidate_token_files() -> list[Path]:
    candidates: list[Path] = []

    # 1) Explicit file path
    explicit = os.environ.get("ILAM_HF_TOKEN_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    # 2) Repo-local: <repo>/ilam_env/.hf_token (works in this repo)
    # In this repo layout, `ilam/` is a top-level package dir.
    try:
        repo_root = Path(__file__).resolve().parent.parent
        candidates.append(repo_root / "ilam_env" / ".hf_token")
    except Exception:
        pass

    # 3) CWD-local: ./ilam_env/.hf_token (useful for notebooks)
    try:
        candidates.append(Path.cwd() / "ilam_env" / ".hf_token")
    except Exception:
        pass

    # De-dup while preserving order
    out: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            pass
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def get_hf_token() -> Optional[str]:
    """
    Return the Hugging Face access token if available, else None.
    """
    global _WARNED_INSECURE_PERMS

    for key in _ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    for token_file in _candidate_token_files():
        if not token_file.exists():
            continue
        try:
            # Warn once if token file is group/world-accessible.
            mode = token_file.stat().st_mode & 0o777
            if (mode & 0o077) and not _WARNED_INSECURE_PERMS:
                print(
                    f"[ilam.hf_auth] Warning: {token_file} permissions are {oct(mode)}. "
                    "Use chmod 600 for safer token storage."
                )
                _WARNED_INSECURE_PERMS = True
        except Exception:
            pass

        try:
            value = token_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if value:
            return value

    return None


def apply_hf_token_env() -> Optional[str]:
    """
    Load token from env/file and mirror into common HF env vars for libraries
    that read from the process environment.
    """
    token = get_hf_token()
    if not token:
        return None

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    os.environ.setdefault("ILAM_HF_TOKEN", token)
    return token

