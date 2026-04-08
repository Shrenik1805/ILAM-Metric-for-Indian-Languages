"""
morph_score.py
--------------
MorphScore: Morpheme-level overlap between hypothesis and reference.

Why this matters:
  - BLEU treats 'kitaab', 'kitaaben', 'kitaabon' as 3 completely different
    tokens. They share the root 'kitaab' (book) and differ only in number/case.
  - Marathi vibhakti (8 case suffixes) causes the same problem.
  - Kannada agglutination stacks 4-6 morphemes; BLEU sees 0% overlap with
    the base form.

Approach:
  1. Tokenize using IndicNLP word tokenizer.
  2. Apply unsupervised morpheme segmentation via IndicNLP.
  3. Fallback: character n-gram bag if IndicNLP morphology unavailable.
  4. Compute F1 over morpheme multisets (precision × recall harmonic mean).

Language-specific notes:
  - Marathi (mr): Heavy vibhakti inflection; 8 case markers.
    Noun 'ghar' (house): ghara, gharala, gharache, gharaat, gharat, gharun ...
  - Kannada (kn): Highly agglutinative.
    Verb root + tense + person + number + gender + honourific can all fuse.
    'maaDuttiddaane' = maaDu + utta + idda + aane (he is doing)
  - Hindi (hi): Moderate inflection; simpler than Marathi.
"""

import re
from collections import Counter
from .script_score import normalize

# ── Marathi vibhakti suffixes (8 cases) ──────────────────────────────────────
MR_VIBHAKTI = [
    "ला", "ने", "चा", "ची", "चे", "त", "ात", "हून", "ून",
    "स", "शी", "ना", "नो", "ो", "ांना", "ांचा", "ांची", "ांचे",
]

# ── Kannada common agglutinative suffixes ─────────────────────────────────────
KN_SUFFIXES = [
    "ನ್ನು", "ಗೆ", "ಇಂದ", "ಅಲ್ಲಿ", "ಉ", "ಅನ್ನು", "ದ", "ತ",
    "ತ್ತಿದ್ದ", "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತಾರೆ", "ತ್ತೇನೆ",
    "ಇರುತ್ತಾನೆ", "ಇರುವ", "ಇರು",
]

# ── Hindi common suffixes ─────────────────────────────────────────────────────
HI_SUFFIXES = [
    "ने", "को", "का", "की", "के", "में", "पर", "से", "तक",
    "ों", "ों को", "ओं", "ें", "ाएं",
]


def _indic_tokenize(text: str, lang: str) -> list:
    """Tokenize using IndicNLP; fallback to whitespace split."""
    try:
        from indicnlp.tokenize import indic_tokenize
        return indic_tokenize.trivial_tokenize(text, lang)
    except Exception:
        return text.split()


def _morph_segment_indic(token: str, lang: str) -> list:
    """
    Attempt IndicNLP unsupervised morpheme segmentation.
    Falls back to suffix-stripping heuristic if unavailable.
    """
    try:
        from indicnlp.morph import unsupervised_morph
        segments = unsupervised_morph.analyze(token, lang)
        if segments and len(segments) > 0:
            return segments
    except Exception:
        pass
    return _suffix_strip(token, lang)


def _suffix_strip(token: str, lang: str) -> list:
    """
    Heuristic suffix stripper for Marathi, Kannada, Hindi.
    Returns [root, suffix] if a known suffix is found, else [token].
    """
    if lang == "mr":
        for sfx in sorted(MR_VIBHAKTI, key=len, reverse=True):
            if token.endswith(sfx) and len(token) > len(sfx) + 1:
                return [token[: -len(sfx)], sfx]
    elif lang == "kn":
        for sfx in sorted(KN_SUFFIXES, key=len, reverse=True):
            if token.endswith(sfx) and len(token) > len(sfx) + 1:
                return [token[: -len(sfx)], sfx]
    elif lang == "hi":
        for sfx in sorted(HI_SUFFIXES, key=len, reverse=True):
            if token.endswith(sfx) and len(token) > len(sfx) + 1:
                return [token[: -len(sfx)], sfx]
    return [token]


def _char_ngrams(token: str, n: int = 3) -> list:
    """Character n-grams as a fallback morpheme proxy."""
    return [token[i: i + n] for i in range(len(token) - n + 1)] or [token]


def _get_morphemes(text: str, lang: str) -> Counter:
    """
    Full morpheme extraction pipeline for a text string.

    Returns a Counter of morphemes (multiset).
    """
    text = normalize(text, lang)
    tokens = _indic_tokenize(text, lang)
    morphemes = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        segs = _morph_segment_indic(tok, lang)
        if len(segs) == 1 and segs[0] == tok:
            # No decomposition found — use char trigrams as proxy
            morphemes.extend(_char_ngrams(tok, n=3))
        else:
            morphemes.extend(segs)
    return Counter(morphemes)


def morph_score(hypothesis: str, reference: str, lang: str) -> float:
    """
    Compute MorphScore: F1 over morpheme multisets.

    F1 = 2 * (precision * recall) / (precision + recall)
    where:
      precision = |hyp_morphemes ∩ ref_morphemes| / |hyp_morphemes|
      recall    = |hyp_morphemes ∩ ref_morphemes| / |ref_morphemes|

    Parameters
    ----------
    hypothesis : str
        Model output.
    reference : str
        Gold reference.
    lang : str
        Language code ('hi', 'mr', 'kn').

    Returns
    -------
    float
        Score in [0, 1].
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0

    hyp_counter = _get_morphemes(hypothesis, lang)
    ref_counter = _get_morphemes(reference, lang)

    if not hyp_counter or not ref_counter:
        return 0.0

    # Multiset intersection
    intersection = sum((hyp_counter & ref_counter).values())
    precision = intersection / sum(hyp_counter.values())
    recall = intersection / sum(ref_counter.values())

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def batch_morph_score(hypotheses: list, references: list, lang: str) -> list:
    """
    Compute MorphScore for a batch of sentence pairs.

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
    return [morph_score(h, r, lang) for h, r in zip(hypotheses, references)]
