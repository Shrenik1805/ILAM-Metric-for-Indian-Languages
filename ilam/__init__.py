"""
ILAM — Indian Language-Aware Metric
=====================================
A linguistically-grounded evaluation metric for Indic NLP tasks.

Supported languages: Hindi (hi), Marathi (mr), Kannada (kn)

Quick start:
    from ilam import ILAM

    scorer = ILAM(lang='mr')
    result = scorer.score(hypothesis="...", reference="...")
    print(result)  # {'ilam': 0.68, 'morph': 0.72, 'sem': 0.64, 'script': 0.91}

    # Batch scoring
    results = scorer.batch_score(hypotheses=[...], references=[...])

    # Corpus-level average
    corpus = scorer.corpus_score(hypotheses=[...], references=[...])
"""

from .metric import ILAM
from .morph_score import morph_score, batch_morph_score
from .sem_score import sem_score, batch_sem_score
from .script_score import script_score, batch_script_score, normalize

__version__ = "0.1.0"
__author__ = "ILAM Research"
__all__ = [
    "ILAM",
    "morph_score",
    "batch_morph_score",
    "sem_score",
    "batch_sem_score",
    "script_score",
    "batch_script_score",
    "normalize",
]
