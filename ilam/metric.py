"""
metric.py
---------
ILAM — Indian Language-Aware Metric
Composite evaluation metric for Indic NLP tasks.

Formula:
    ILAM = α × MorphScore + β × SemScore + γ × ScriptScore

Default weights (language-adaptive presets shipped with this package):
    Hindi (hi)   : α=0.35, β=0.45, γ=0.20
    Marathi (mr) : α=0.45, β=0.35, γ=0.20  (vibhakti more important)
    Kannada (kn) : α=0.45, β=0.35, γ=0.20  (agglutination more important)

The weights can be overridden at instantiation time.

Usage
-----
    from ilam import ILAM

    scorer = ILAM(lang='mr')
    result = scorer.score("माझे घर", "माझ्या घरात")
    # {'ilam': 0.64, 'morph': 0.71, 'sem': 0.58, 'script': 0.62}

    # Batch
    results = scorer.batch_score(hypotheses=[...], references=[...])

    # Single float
    s = scorer.score_value("माझे घर", "माझ्या घरात")
"""

from .morph_score import morph_score, batch_morph_score
from .sem_score import sem_score, batch_sem_score
from .script_score import script_score, batch_script_score, normalize

# ── Default language-adaptive weights ────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "hi": {"alpha": 0.35, "beta": 0.45, "gamma": 0.20},
    "mr": {"alpha": 0.45, "beta": 0.35, "gamma": 0.20},
    "kn": {"alpha": 0.45, "beta": 0.35, "gamma": 0.20},
    # Defaults for unseen languages
    "default": {"alpha": 0.40, "beta": 0.40, "gamma": 0.20},
}

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
}


class ILAM:
    """
    Indian Language-Aware Metric.

    Parameters
    ----------
    lang : str
        ISO 639-1 language code for the TARGET language being evaluated.
        Supported: 'hi' (Hindi), 'mr' (Marathi), 'kn' (Kannada).
    alpha : float, optional
        Weight for MorphScore (default: language-adaptive preset).
    beta : float, optional
        Weight for SemScore (default: language-adaptive preset).
    gamma : float, optional
        Weight for ScriptScore (default: language-adaptive preset).
    sem_model : str, optional
        HuggingFace model for SemScore (default: 'google/muril-base-cased').
        Set to None to force character-level fallback.
    verbose : bool
        Print progress during batch scoring (default: False).
    """

    def __init__(
        self,
        lang: str = "hi",
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        sem_model: str = "google/muril-base-cased",
        verbose: bool = False,
    ):
        self.lang = lang
        self.sem_model = sem_model
        self.verbose = verbose

        # Load language-adaptive defaults
        presets = DEFAULT_WEIGHTS.get(lang, DEFAULT_WEIGHTS["default"])
        self.alpha = alpha if alpha is not None else presets["alpha"]
        self.beta = beta if beta is not None else presets["beta"]
        self.gamma = gamma if gamma is not None else presets["gamma"]

        # Normalise weights to sum to 1.0
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

        lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
        if verbose:
            print(
                f"[ILAM] Language: {lang_name} ({lang}) | "
                f"weights → morph={self.alpha:.2f}, "
                f"sem={self.beta:.2f}, "
                f"script={self.gamma:.2f}"
            )

    # ── Single sentence ──────────────────────────────────────────────────────

    def score(self, hypothesis: str, reference: str) -> dict:
        """
        Score a single hypothesis against a reference.

        Returns
        -------
        dict with keys: 'ilam', 'morph', 'sem', 'script'
        """
        ms = morph_score(hypothesis, reference, self.lang)
        ss = sem_score(hypothesis, reference, self.lang, self.sem_model)
        sc = script_score(hypothesis, reference, self.lang)

        composite = round(
            self.alpha * ms + self.beta * ss + self.gamma * sc, 4
        )

        return {
            "ilam": composite,
            "morph": ms,
            "sem": ss,
            "script": sc,
        }

    def score_value(self, hypothesis: str, reference: str) -> float:
        """Return only the composite ILAM float score."""
        return self.score(hypothesis, reference)["ilam"]

    # ── Batch ────────────────────────────────────────────────────────────────

    def batch_score(
        self,
        hypotheses: list,
        references: list,
        batch_size: int = 32,
    ) -> list:
        """
        Score a list of hypothesis-reference pairs.

        Parameters
        ----------
        hypotheses : list of str
        references  : list of str
        batch_size  : int  (for SemScore batching)

        Returns
        -------
        list of dict, each with keys: 'ilam', 'morph', 'sem', 'script'
        """
        assert len(hypotheses) == len(references), "Length mismatch"
        n = len(hypotheses)

        if self.verbose:
            print(f"[ILAM] Scoring {n} sentence pairs for lang={self.lang} ...")

        # Compute sub-scores
        morph_scores = batch_morph_score(hypotheses, references, self.lang)
        sem_scores = batch_sem_score(
            hypotheses, references, self.lang, self.sem_model, batch_size
        )
        script_scores = batch_script_score(hypotheses, references, self.lang)

        results = []
        for ms, ss, sc in zip(morph_scores, sem_scores, script_scores):
            composite = round(self.alpha * ms + self.beta * ss + self.gamma * sc, 4)
            results.append({"ilam": composite, "morph": ms, "sem": ss, "script": sc})

        if self.verbose:
            avg = sum(r["ilam"] for r in results) / n
            print(f"[ILAM] Done. Average ILAM score: {avg:.4f}")

        return results

    def corpus_score(self, hypotheses: list, references: list) -> dict:
        """
        Compute corpus-level ILAM scores (average over all sentence pairs).

        Returns
        -------
        dict with keys: 'ilam', 'morph', 'sem', 'script'
            Each value is the mean over the corpus.
        """
        results = self.batch_score(hypotheses, references)
        keys = ["ilam", "morph", "sem", "script"]
        n = len(results)
        return {
            k: round(sum(r[k] for r in results) / n, 4)
            for k in keys
        }

    def __repr__(self):
        return (
            f"ILAM(lang={self.lang}, "
            f"α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f})"
        )
