"""
script_score.py
---------------
ScriptScore: Unicode normalization + character-level chrF similarity.

Handles:
  - Devanagari normalization (Hindi, Marathi) — multiple valid Unicode
    sequences for the same glyph, nukta placement, chandrabindu variants.
  - Kannada script normalization — anusvara/visarga variants, vowel signs.
  - Returns a [0,1] score: 1 = identical after normalization.

Used both as a preprocessing step (normalize before other sub-scores)
and as a standalone sub-score in the ILAM composite.
"""

import unicodedata
import re
from collections import Counter

try:
    from sacrebleu.metrics import CHRF
    _CHRF = CHRF(char_order=6, word_order=0, beta=2)
except Exception:
    CHRF = None
    _CHRF = None


# ── Language → script Unicode block ──────────────────────────────────────────
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari
    "mr": (0x0900, 0x097F),   # Devanagari
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "bn": (0x0980, 0x09FF),   # Bengali (future)
    "te": (0x0C00, 0x0C7F),   # Telugu (future)
}

# ── Devanagari-specific normalisation patterns ────────────────────────────────
# Map alternate encodings to canonical forms
DEVA_NORM = [
    ("\u0928\u093C", "\u0929"),   # na + nukta → ṉa (NNNA)
    ("\u0930\u093C", "\u0931"),   # ra + nukta → ṟa (RRA)
    ("\u0933\u093C", "\u0934"),   # la + nukta → ḷa (LLLA)
    ("\u0915\u093C", "\u0958"),   # ka + nukta → qa
    ("\u0916\u093C", "\u0959"),   # kha + nukta → ḵa
    ("\u0917\u093C", "\u095A"),   # ga + nukta → ġa
    ("\u091C\u093C", "\u095B"),   # ja + nukta → za
    ("\u0921\u093C", "\u095C"),   # dda + nukta → ṛa
    ("\u0922\u093C", "\u095D"),   # ddha + nukta → ṛha
    ("\u092B\u093C", "\u095E"),   # pha + nukta → fa
    ("\u092F\u093C", "\u095F"),   # ya + nukta → yya
]

# Kannada-specific normalisation
KANNA_NORM = [
    ("\u0CB0\u0CBC", "\u0CDE"),   # ra + nukta → RRA
]


def _apply_norm_map(text: str, norm_map: list) -> str:
    for src, tgt in norm_map:
        text = text.replace(src, tgt)
    return text


def _char_f_score(hypothesis: str, reference: str, n: int = 6) -> float:
    def grams(text: str) -> Counter:
        text = text.strip()
        if len(text) < n:
            return Counter([text]) if text else Counter()
        return Counter(text[i: i + n] for i in range(len(text) - n + 1))

    hyp_grams = grams(hypothesis)
    ref_grams = grams(reference)
    if not hyp_grams or not ref_grams:
        return 0.0

    overlap = sum((hyp_grams & ref_grams).values())
    precision = overlap / max(1, sum(hyp_grams.values()))
    recall = overlap / max(1, sum(ref_grams.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def normalize(text: str, lang: str) -> str:
    """
    Normalize an Indic text string to its canonical Unicode form.

    Steps:
      1. NFC normalization (canonical decomposition + composition)
      2. Language-specific glyph normalization
      3. Collapse multiple whitespace
      4. Strip leading/trailing whitespace

    Parameters
    ----------
    text : str
        Raw input text.
    lang : str
        ISO 639-1 language code ('hi', 'mr', 'kn').

    Returns
    -------
    str
        Normalized text.
    """
    # Step 1: Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Step 2: Language-specific
    if lang in ("hi", "mr"):
        text = _apply_norm_map(text, DEVA_NORM)
    elif lang == "kn":
        text = _apply_norm_map(text, KANNA_NORM)

    # Step 3: IndicNLP normalizer (if available)
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        factory = IndicNormalizerFactory()
        # Map lang codes to IndicNLP codes
        lang_map = {"hi": "hi", "mr": "mr", "kn": "kn", "bn": "bn", "te": "te"}
        if lang in lang_map:
            normalizer = factory.get_normalizer(lang_map[lang])
            text = normalizer.normalize(text)
    except Exception:
        pass  # graceful fallback if IndicNLP not available

    # Step 4: Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def script_score(hypothesis: str, reference: str, lang: str) -> float:
    """
    Compute ScriptScore between hypothesis and reference.

    Both strings are first normalized, then character-level chrF (order=6)
    is computed as the script fidelity score.

    Parameters
    ----------
    hypothesis : str
        Model output text.
    reference : str
        Gold reference text.
    lang : str
        ISO 639-1 language code.

    Returns
    -------
    float
        Score in [0, 1].
    """
    hyp_norm = normalize(hypothesis, lang)
    ref_norm = normalize(reference, lang)

    if not hyp_norm.strip() or not ref_norm.strip():
        return 0.0

    if _CHRF is not None:
        result = _CHRF.sentence_score(hyp_norm, [ref_norm])
        return round(result.score / 100.0, 4)

    # Fallback when sacrebleu is unavailable.
    return round(_char_f_score(hyp_norm, ref_norm, n=6), 4)


def batch_script_score(hypotheses: list, references: list, lang: str) -> list:
    """
    Compute ScriptScore for a batch of sentence pairs.

    Parameters
    ----------
    hypotheses : list of str
    references  : list of str
    lang        : str

    Returns
    -------
    list of float
    """
    assert len(hypotheses) == len(references), "Length mismatch"
    return [script_score(h, r, lang) for h, r in zip(hypotheses, references)]
