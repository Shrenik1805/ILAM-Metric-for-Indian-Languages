"""
Shared Hugging Face token loader for this project.

Priority:
1. Environment variables (ILAM_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_TOKEN)
2. Token file at ilam_env/.hf_token
"""

from pathlib import Path
import os
from typing import Optional


_ENV_KEYS = (
    "ILAM_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)

_TOKEN_FILE = Path(__file__).resolve().parent / "ilam_env" / ".hf_token"
_WARNED_INSECURE_PERMS = False


def get_hf_token() -> Optional[str]:
    global _WARNED_INSECURE_PERMS
    for key in _ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    if _TOKEN_FILE.exists():
        try:
            # Warn once if token file is group/world-accessible.
            mode = _TOKEN_FILE.stat().st_mode & 0o777
            if (mode & 0o077) and not _WARNED_INSECURE_PERMS:
                print(
                    f"[hf_auth] Warning: {_TOKEN_FILE} permissions are {oct(mode)}. "
                    "Use chmod 600 for safer token storage."
                )
                _WARNED_INSECURE_PERMS = True
        except Exception:
            pass
        value = _TOKEN_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def apply_hf_token_env() -> Optional[str]:
    """
    Load token from env/file and mirror into common HF env vars for libs that
    read from process environment.
    """
    token = get_hf_token()
    if not token:
        return None

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    os.environ.setdefault("ILAM_HF_TOKEN", token)
    return token
