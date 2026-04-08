"""
sem_score.py
------------
SemScore: Semantic similarity via MuRIL embeddings or char n-gram fallback.

Environment detection:
  - GPU (CUDA) available  → uses google/muril-base-cased
  - CPU / Mac / no torch  → uses character trigram cosine (safe, no crash)

The char n-gram fallback is effective for Indic scripts because morphological
variants share large character n-gram overlap, capturing partial semantic
similarity without any model loading.

To force MuRIL on a GPU machine:
    sem_score(hyp, ref, lang, force_model=True)
"""

import math
from collections import Counter
from .script_score import normalize
from .hf_auth import apply_hf_token_env

_MODEL_CACHE = {}
_TORCH_AVAILABLE = None   # cached after first check
_WARNED = set()


def _warn_once(key: str, msg: str):
    if key in _WARNED:
        return
    print(msg)
    _WARNED.add(key)


def _check_torch_gpu() -> bool:
    """
    Check if torch with CUDA is available.
    Returns False on import error, MPS-only, or CPU-only.
    Caches result so torch is only imported once.
    """
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        # Use subprocess to avoid crashing the main process if torch is broken
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print('1' if torch.cuda.is_available() else '0')"],
            capture_output=True, text=True, timeout=10
        )
        _TORCH_AVAILABLE = result.returncode == 0 and result.stdout.strip() == "1"
    except Exception:
        _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _load_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from transformers import AutoTokenizer, AutoModel
    import torch
    hf_token = apply_hf_token_env()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token)
    model.eval()
    _MODEL_CACHE[model_name] = (tokenizer, model, torch)
    return tokenizer, model, torch


def _mean_pool(token_embeddings, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def _embed_muril(texts: list, model_name: str) -> list:
    tokenizer, model, torch = _load_model(model_name)
    embeddings = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            out = model(**enc)
            emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeddings.append(emb.squeeze().numpy())
    return embeddings


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Character n-gram cosine (pure Python, no torch, no crash) ────────────────

def _char_ngram_vector(text: str, n: int = 3) -> Counter:
    text = text.strip()
    return Counter(text[i: i + n] for i in range(len(text) - n + 1))


def _char_cosine(text_a: str, text_b: str, n: int = 3) -> float:
    va = _char_ngram_vector(text_a, n)
    vb = _char_ngram_vector(text_b, n)
    if not va or not vb:
        return 0.0
    keys = set(va) | set(vb)
    dot = sum(va[k] * vb[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in va.values()))
    norm_b = math.sqrt(sum(v * v for v in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Public API ────────────────────────────────────────────────────────────────

def sem_score(
    hypothesis: str,
    reference: str,
    lang: str,
    model_name: str = "google/muril-base-cased",
    force_model: bool = False,
) -> float:
    """
    Compute SemScore between hypothesis and reference.

    Automatically uses char n-gram cosine on CPU/Mac to prevent crashes.
    Pass force_model=True to load MuRIL even on CPU (slow, may crash on Mac).

    Returns float in [0, 1].
    """
    hyp = normalize(hypothesis, lang)
    ref = normalize(reference, lang)

    if not hyp.strip() or not ref.strip():
        return 0.0

    use_model = force_model or (model_name and _check_torch_gpu())

    if use_model:
        try:
            embs = _embed_muril([hyp, ref], model_name)
            score = _cosine(embs[0], embs[1])
            return round(max(0.0, min(1.0, score)), 4)
        except Exception as e:
            _warn_once("sem_score_model_fallback", f"[SemScore] Warning: model path failed, using char fallback ({e})")

    return round(_char_cosine(hyp, ref, n=3), 4)


def batch_sem_score(
    hypotheses: list,
    references: list,
    lang: str,
    model_name: str = "google/muril-base-cased",
    batch_size: int = 32,
    force_model: bool = False,
) -> list:
    """
    Batch SemScore. GPU-safe: uses char n-gram on CPU/Mac automatically.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Length mismatch: hypotheses and references must be equal length")

    hyps_norm = [normalize(h, lang) for h in hypotheses]
    refs_norm = [normalize(r, lang) for r in references]

    use_model = force_model or (model_name and _check_torch_gpu())

    if use_model:
        try:
            import torch
            tokenizer, model, torch = _load_model(model_name)
            all_texts = hyps_norm + refs_norm
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(all_texts), batch_size):
                    batch = all_texts[i: i + batch_size]
                    enc = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128)
                    out = model(**enc)
                    embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                    all_embs.extend(embs.numpy())
            n = len(hypotheses)
            return [
                round(max(0.0, min(1.0, _cosine(all_embs[i], all_embs[n + i]))), 4)
                for i in range(n)
            ]
        except Exception as e:
            _warn_once("batch_sem_score_model_fallback", f"[SemScore] Warning: batch model path failed, using char fallback ({e})")

    # Safe fallback — works on Mac, CPU, no GPU, broken torch
    return [round(_char_cosine(h, r, n=3), 4) for h, r in zip(hyps_norm, refs_norm)]
