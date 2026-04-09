# ILAM — Indian Language-Aware Metric

**ILAM** is a linguistically-grounded evaluation metric for Indic NLP tasks, with a companion cross-lingual transfer pipeline for Hindi → Marathi and Hindi → Kannada.

---

## Motivation

Standard metrics like BLEU fail for Indian languages because:

| Problem | Example | BLEU Score | ILAM Score |
|---|---|---|---|
| Morphological variants | `kitaab` vs `kitaabon` (same root) | 0% overlap | Partial credit via MorphScore |
| Marathi vibhakti | `ghara` vs `gharaat` (at home) | 0% | Root matched |
| Kannada agglutination | `maaDuttiddaane` vs `maaDu` | 0% | Root matched |
| Semantic paraphrase | Different words, same meaning | Low | High via SemScore |

---

## Architecture

```
ILAM = α × MorphScore + β × SemScore + γ × ScriptScore
```

| Sub-score | What it measures | Default weight |
|---|---|---|
| **MorphScore** | Morpheme-level F1 overlap (IndicNLP) | 0.40–0.45 |
| **SemScore** | Cosine similarity via MuRIL embeddings | 0.35–0.45 |
| **ScriptScore** | Unicode normalization + chrF fidelity | 0.20 |

Weights are **language-adaptive**:
- **Marathi / Kannada**: MorphScore weight increased (heavy morphology)
- **Hindi**: SemScore weight increased (simpler morphology)

---

## Installation

```bash
# Install package in current environment
pip install .

# Editable install for development
pip install -e .

# If your environment blocks network during build, use:
pip install -e . --no-build-isolation

# Core (CPU-only, no GPU needed)
pip install -r requirements.txt

# Dev tooling (lint/type/test/audit)
pip install -r requirements-dev.txt

# Translation + Flores (IndicTrans2)
pip install -r requirements-transfer.txt

# With MuRIL semantic scoring (recommended, needs GPU)
pip install -r requirements.txt
pip install transformers torch sentencepiece

# Full pipeline with IndicTrans2 transfer (recommended)
pip install -r requirements-transfer.txt

# Extras via package metadata
pip install ".[dev]"
pip install ".[gpu]"
pip install ".[transfer]"
```

---

## Hugging Face Token (Private)

Store your token privately in `ilam_env/.hf_token` (permissions `600`):

```bash
./scripts/set_hf_token.sh
```

To rotate/delete old cached HF tokens and set a new one:

```bash
./scripts/rotate_hf_token.sh
```

All project code reads the token through `ilam.hf_auth`.

---

## Dev Workflow

```bash
# Lint + format check
make lint

# Type checks
make typecheck

# Unit tests
make test

# Dependency audit
make audit
```

---

## Quick Start

```python
from ilam import ILAM

# Score a single sentence pair (Marathi)
scorer = ILAM(lang='mr')
result = scorer.score(
    hypothesis="मी रोज सकाळी उद्यानात फिरतो.",
    reference="मी दररोज सकाळी बागेत फेरफटका मारतो."
)
print(result)
# {'ilam': 0.68, 'morph': 0.71, 'sem': 0.64, 'script': 0.89}

# Kannada
scorer_kn = ILAM(lang='kn')
result = scorer_kn.score(
    hypothesis="ಭಾರತ ವೈವಿಧ್ಯತೆಯಿಂದ ತುಂಬಿದ ದೇಶ.",
    reference="ಭಾರತ ವಿವಿಧತೆಗಳಿಂದ ತುಂಬಿರುವ ದೇಶ."
)

# Batch scoring
results = scorer.batch_score(
    hypotheses=["...", "...", "..."],
    references=["...", "...", "..."]
)

# Corpus-level average
corpus = scorer.corpus_score(hypotheses=[...], references=[...])
```

---

## Supported Languages

| Code | Language | Family | Script |
|---|---|---|---|
| `hi` | Hindi | Indo-Aryan | Devanagari |
| `mr` | Marathi | Indo-Aryan | Devanagari |
| `kn` | Kannada | Dravidian | Kannada |

---

## Running the Full Pipeline

### Demo mode (no GPU needed)
```bash
python run_all.py --demo
```
Outputs are written to `results/`.

### Full pipeline with IndicTrans2 translation (GPU required)
```bash
python run_all.py --translate --src_lang hi --tgt_langs mr kn --max_samples 200
```
Outputs are written to `results_flores/` by default.

### Score existing translations (no re-translation)
If you already have `data/translations/*.json` (e.g., from a previous run), you can skip the translation step:
```bash
python run_all.py --translate --skip_translate_step
```

### Individual steps
```bash
# Baselines only
python experiments/run_baselines.py --demo

# ILAM scoring only
python experiments/run_ilam.py --demo

# Correlation analysis
python experiments/correlation.py --demo
```

---

## Output Files

After running `run_all.py`, the `results/` directory contains:

| File | Contents |
|---|---|
| `baseline_scores.csv` | BLEU, chrF, chrF++ corpus scores |
| `ilam_scores_mr.csv` | Sentence-level ILAM scores for Marathi |
| `ilam_scores_kn.csv` | Sentence-level ILAM scores for Kannada |
| `ilam_summary.csv` | Corpus-level ILAM averages |
| `correlation_table.csv` | Pearson/Spearman/Kendall vs human judgments |
| `correlation_table.tex` | LaTeX table for paper |
| `correlation_report.txt` | Full text report |

---

## Cross-Lingual Transfer

### Zero-Shot (no fine-tuning)
```python
from transfer import IndicTranslator

translator = IndicTranslator()
results = translator.translate_flores200(
    src_lang="hi",
    tgt_langs=["mr", "kn"],
    max_samples=200,
    save_dir="data/translations"
)
```

### Single sentence
```python
out = translator.translate("मैं घर जाता हूँ", src_lang="hi", tgt_lang="mr")
```

---

## Running Tests

```bash
python tests/test_ilam.py

# Or with pytest
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
ilam/
├── ilam/
│   ├── __init__.py          # Package exports
│   ├── morph_score.py       # Morpheme-level F1 (IndicNLP)
│   ├── sem_score.py         # MuRIL semantic similarity
│   ├── script_score.py      # Unicode normalization + chrF
│   └── metric.py            # Composite ILAM scorer
├── transfer/
│   ├── __init__.py
│   └── translate.py         # IndicTrans2 zero-shot pipeline
├── experiments/
│   ├── run_baselines.py     # BLEU, chrF, chrF++ baselines
│   ├── run_ilam.py          # ILAM scoring experiments
│   └── correlation.py       # Correlation vs human judgments
├── tests/
│   └── test_ilam.py         # 28 unit tests
├── data/                    # Datasets (gitignored)
├── results/                 # Generated outputs (gitignored)
├── run_all.py               # End-to-end pipeline runner
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Why MorphScore uses IndicNLP + suffix stripping fallback:**
IndicNLP's unsupervised morpheme segmenter works well for Hindi and Marathi but is weaker for Kannada's deep agglutination. The suffix-stripping fallback with known Kannada suffixes handles the most common cases.

**Why char n-gram cosine as SemScore fallback:**
When MuRIL is unavailable (no GPU), character n-gram cosine similarity over Indic text is surprisingly effective because morphological variants share many character n-grams.

**Why weights are normalized:**
Custom weight overrides are always normalized to sum to 1.0, so `ILAM(alpha=2, beta=1, gamma=1)` gives `alpha=0.5, beta=0.25, gamma=0.25`.

---

## Target Venues

- **ICON 2025** — International Conference on Natural Language Processing (India)
- **WAT 2025** — Workshop on Asian Translation (co-located with ACL)
- **LoResMT @ ACL/EMNLP** — Low-Resource Machine Translation

---

## Citation

```bibtex
@inproceedings{ilam2025,
  title     = {ILAM: An Indian Language-Aware Metric for Evaluating
               Cross-Lingual Transfer in Low-Resource Indic Languages},
  booktitle = {Proceedings of ICON 2025},
  year      = {2025},
}
```

---

## License

NA
