"""
test_ilam.py
------------
Unit tests for all ILAM sub-scores and the composite metric.

Run: python -m pytest tests/test_ilam.py -v
Or:  python tests/test_ilam.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ilam.script_score import normalize, script_score
from ilam.morph_score import morph_score, _suffix_strip
from ilam.sem_score import sem_score, _char_cosine
from ilam.metric import ILAM


# ── ScriptScore Tests ────────────────────────────────────────────────────────

def test_normalize_hindi():
    text = "मैं घर जाता हूँ"
    result = normalize(text, "hi")
    assert isinstance(result, str)
    assert len(result) > 0

def test_normalize_marathi():
    text = "माझे घर"
    result = normalize(text, "mr")
    assert isinstance(result, str)

def test_normalize_kannada():
    text = "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ"
    result = normalize(text, "kn")
    assert isinstance(result, str)

def test_script_score_identical():
    # Identical strings → score should be close to 1.0
    text = "भारत एक महान देश है"
    s = script_score(text, text, "hi")
    assert s > 0.95, f"Expected >0.95 for identical, got {s}"

def test_script_score_empty():
    s = script_score("", "भारत", "hi")
    assert s == 0.0

def test_script_score_different_scripts():
    # Hindi vs Kannada: should be low
    s = script_score("मैं घर जाता हूँ", "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ", "hi")
    assert s < 0.5, f"Expected <0.5 for different scripts, got {s}"


# ── MorphScore Tests ─────────────────────────────────────────────────────────

def test_morph_score_identical():
    text = "भारत एक महान देश है"
    s = morph_score(text, text, "hi")
    assert s > 0.9, f"Expected >0.9 for identical, got {s}"

def test_morph_score_range():
    hyp = "मी दररोज सकाळी बागेत फेरफटका मारतो"
    ref = "मी रोज सकाळी उद्यानात फिरतो"
    s = morph_score(hyp, ref, "mr")
    assert 0.0 <= s <= 1.0, f"Out of range: {s}"

def test_morph_score_empty():
    s = morph_score("", "भारत", "hi")
    assert s == 0.0

def test_suffix_strip_marathi():
    # "घरात" → root "घर" + suffix "ात"
    result = _suffix_strip("घरात", "mr")
    assert len(result) >= 1

def test_suffix_strip_kannada():
    # Kannada word with suffix
    result = _suffix_strip("ದೇಶದಲ್ಲಿ", "kn")
    assert len(result) >= 1

def test_morph_score_similar_marathi():
    # Two ways to say the same thing — should score reasonably
    hyp = "शिक्षण हा समाजाचा पाया आहे"
    ref = "शिक्षण हा समाजाचा आधारस्तंभ आहे"
    s = morph_score(hyp, ref, "mr")
    assert s > 0.3, f"Expected >0.3 for similar sentences, got {s}"


# ── SemScore Tests ────────────────────────────────────────────────────────────

def test_char_cosine_identical():
    text = "भारत एक महान देश है"
    s = _char_cosine(text, text)
    assert s > 0.99

def test_char_cosine_empty():
    s = _char_cosine("", "text")
    assert s == 0.0

def test_char_cosine_range():
    a = "मैं घर जाता हूँ"
    b = "वह स्कूल जाती है"
    s = _char_cosine(a, b)
    assert 0.0 <= s <= 1.0

def test_sem_score_fallback():
    # sem_score should fall back to char cosine if MuRIL not available
    hyp = "भारत एक महान देश है"
    ref = "भारत एक विशाल देश है"
    s = sem_score(hyp, ref, "hi", model_name=None)
    assert 0.0 <= s <= 1.0

def test_sem_score_empty():
    s = sem_score("", "text", "hi", model_name=None)
    assert s == 0.0


# ── ILAM Composite Tests ──────────────────────────────────────────────────────

def test_ilam_init_hindi():
    scorer = ILAM(lang="hi")
    assert scorer.lang == "hi"
    assert abs(scorer.alpha + scorer.beta + scorer.gamma - 1.0) < 1e-6

def test_ilam_init_marathi():
    scorer = ILAM(lang="mr")
    assert scorer.lang == "mr"
    # Marathi should weight morph more
    assert scorer.alpha >= scorer.beta

def test_ilam_init_kannada():
    scorer = ILAM(lang="kn")
    assert scorer.lang == "kn"

def test_ilam_score_returns_dict():
    scorer = ILAM(lang="mr")
    result = scorer.score("माझे घर", "माझ्या घरात")
    assert "ilam" in result
    assert "morph" in result
    assert "sem" in result
    assert "script" in result

def test_ilam_score_range():
    scorer = ILAM(lang="mr")
    result = scorer.score("माझे घर", "माझ्या घरात")
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"Sub-score {k}={v} out of [0,1]"

def test_ilam_identical_is_high():
    scorer = ILAM(lang="hi")
    text = "भारत एक महान देश है"
    result = scorer.score(text, text)
    assert result["ilam"] > 0.85, f"Identical text ILAM={result['ilam']}, expected >0.85"

def test_ilam_score_value():
    scorer = ILAM(lang="kn")
    text = "ಭಾರತ ವೈವಿಧ್ಯತೆಯ ದೇಶ"
    s = scorer.score_value(text, text)
    assert isinstance(s, float)
    assert s > 0.85

def test_ilam_batch_score():
    scorer = ILAM(lang="mr")
    hyps = ["माझे घर", "भारत एक देश आहे"]
    refs = ["माझ्या घरात", "भारत हा देश आहे"]
    results = scorer.batch_score(hyps, refs)
    assert len(results) == 2
    for r in results:
        assert "ilam" in r

def test_ilam_corpus_score():
    scorer = ILAM(lang="kn")
    hyps = ["ನಾನು ಹೋಗುತ್ತೇನೆ", "ಭಾರತ ದೇಶ"]
    refs = ["ನಾನು ಹೋಗುತ್ತಿದ್ದೇನೆ", "ಭಾರತ ರಾಷ್ಟ್ರ"]
    corpus = scorer.corpus_score(hyps, refs)
    assert "ilam" in corpus
    assert 0.0 <= corpus["ilam"] <= 1.0

def test_ilam_custom_weights():
    scorer = ILAM(lang="hi", alpha=0.5, beta=0.3, gamma=0.2)
    assert abs(scorer.alpha + scorer.beta + scorer.gamma - 1.0) < 1e-6

def test_ilam_repr():
    scorer = ILAM(lang="mr")
    r = repr(scorer)
    assert "ILAM" in r
    assert "mr" in r


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (test_normalize_hindi, "normalize_hindi"),
        (test_normalize_marathi, "normalize_marathi"),
        (test_normalize_kannada, "normalize_kannada"),
        (test_script_score_identical, "script_score_identical"),
        (test_script_score_empty, "script_score_empty"),
        (test_script_score_different_scripts, "script_score_different_scripts"),
        (test_morph_score_identical, "morph_score_identical"),
        (test_morph_score_range, "morph_score_range"),
        (test_morph_score_empty, "morph_score_empty"),
        (test_suffix_strip_marathi, "suffix_strip_marathi"),
        (test_suffix_strip_kannada, "suffix_strip_kannada"),
        (test_morph_score_similar_marathi, "morph_score_similar_marathi"),
        (test_char_cosine_identical, "char_cosine_identical"),
        (test_char_cosine_empty, "char_cosine_empty"),
        (test_char_cosine_range, "char_cosine_range"),
        (test_sem_score_fallback, "sem_score_fallback"),
        (test_sem_score_empty, "sem_score_empty"),
        (test_ilam_init_hindi, "ilam_init_hindi"),
        (test_ilam_init_marathi, "ilam_init_marathi"),
        (test_ilam_init_kannada, "ilam_init_kannada"),
        (test_ilam_score_returns_dict, "ilam_score_returns_dict"),
        (test_ilam_score_range, "ilam_score_range"),
        (test_ilam_identical_is_high, "ilam_identical_is_high"),
        (test_ilam_score_value, "ilam_score_value"),
        (test_ilam_batch_score, "ilam_batch_score"),
        (test_ilam_corpus_score, "ilam_corpus_score"),
        (test_ilam_custom_weights, "ilam_custom_weights"),
        (test_ilam_repr, "ilam_repr"),
    ]

    passed = 0
    failed = 0
    for fn, name in tests:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
