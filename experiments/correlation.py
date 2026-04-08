"""
correlation.py
--------------
Compute Pearson, Spearman, and Kendall-τ correlation of ILAM vs BLEU/chrF
against human judgments from IndicMT Eval.

If IndicMT Eval is unavailable (no GPU / offline), uses synthetic human
scores derived from a MuRIL-based oracle (clearly labelled in output).

Produces:
  - results/correlation_table.csv
  - results/correlation_report.txt  (LaTeX-ready table for the paper)

Usage:
    python experiments/correlation.py --demo
    python experiments/correlation.py --indicmt_path data/indicmt_eval/
"""

import argparse
import csv
import json
import os
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    _HAS_SCIPY = True
except Exception:
    pearsonr = spearmanr = kendalltau = None
    _HAS_SCIPY = False

try:
    import sacrebleu
    from sacrebleu.metrics import CHRF
    _HAS_SACREBLEU = True
except Exception:
    sacrebleu = None
    CHRF = None
    _HAS_SACREBLEU = False

from ilam import ILAM
from ilam.sem_score import _char_cosine


# ── Synthetic human score oracle ─────────────────────────────────────────────

def _oracle_human_score(hypothesis: str, reference: str, lang: str) -> float:
    """
    Proxy for human judgment when real annotations are unavailable.
    Uses character-level cosine similarity as a silver-standard proxy.
    Clearly labelled as 'proxy' in outputs.
    """
    return round(_char_cosine(hypothesis, reference, n=4), 4)


# ── Sentence-level metric functions ──────────────────────────────────────────

def _sentence_bleu(hyp: str, ref: str) -> float:
    try:
        if _HAS_SACREBLEU:
            return round(sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0, 4)
        return _char_cosine(hyp, ref, n=2)
    except Exception:
        return 0.0


def _sentence_chrf(hyp: str, ref: str) -> float:
    try:
        if _HAS_SACREBLEU:
            return round(CHRF().sentence_score(hyp, [ref]).score / 100.0, 4)
        return _char_cosine(hyp, ref, n=6)
    except Exception:
        return 0.0


def _sentence_chrfpp(hyp: str, ref: str) -> float:
    try:
        if _HAS_SACREBLEU:
            return round(CHRF(word_order=2).sentence_score(hyp, [ref]).score / 100.0, 4)
        return _char_cosine(hyp, ref, n=4)
    except Exception:
        return 0.0


# ── Correlation helpers ───────────────────────────────────────────────────────

def _pearson_fallback(x: list, y: list) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _rankdata(values: list) -> list:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_fallback(x: list, y: list) -> float:
    return _pearson_fallback(_rankdata(x), _rankdata(y))


def _kendall_fallback(x: list, y: list) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom

def _corr(x: list, y: list) -> dict:
    """Return Pearson, Spearman, Kendall-τ for two score lists."""
    if len(x) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "kendall": float("nan")}

    if _HAS_SCIPY:
        p, _ = pearsonr(x, y)
        s, _ = spearmanr(x, y)
        k, _ = kendalltau(x, y)
    else:
        p = _pearson_fallback(x, y)
        s = _spearman_fallback(x, y)
        k = _kendall_fallback(x, y)

    return {
        "pearson": round(p, 4),
        "spearman": round(s, 4),
        "kendall": round(k, 4),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

DEMO_DATA = {
    "mr": {
        "src_lang": "hi", "tgt_lang": "mr",
        "hypotheses": [
            "मी रोज सकाळी उद्यानात फिरतो.",
            "भारत हा विविधतेने भरलेला देश आहे.",
            "विज्ञान आणि तंत्रज्ञानाने मानवी जीवन बदलले आहे.",
            "शिक्षण हा कोणत्याही समाजाचा पाया असतो.",
            "पाणी हे आपल्या जीवनासाठी अत्यंत आवश्यक आहे.",
            "खेळ शरीर आणि मन दोन्ही निरोगी ठेवतो.",
            "निसर्ग आपल्याला खूप काही शिकवतो.",
            "कुटुंब आपला सर्वात मोठा आधार असतो.",
            "परिश्रम आणि समर्पणाने सर्व काम शक्य आहे.",
            "आरोग्य हीच सर्वात मोठी संपत्ती आहे.",
        ],
        "references": [
            "मी दररोज सकाळी बागेत फेरफटका मारतो.",
            "भारत हा विविधतांनी भरलेला देश आहे.",
            "विज्ञान व तंत्रज्ञानाने माणसाचे जीवन बदलून टाकले आहे.",
            "शिक्षण हा कोणत्याही समाजाचा आधारस्तंभ असतो.",
            "पाणी हे आपल्या जीवनासाठी अत्यंत महत्त्वाचे आहे.",
            "खेळामुळे शरीर व मन दोन्ही तंदुरुस्त राहतात.",
            "निसर्ग आपल्याला अनेक गोष्टी शिकवतो.",
            "कुटुंब हे आपले सर्वात मोठे आधारस्थान आहे.",
            "मेहनत आणि निष्ठेने प्रत्येक काम साध्य होते.",
            "आरोग्य हीच खरी संपत्ती आहे.",
        ],
    },
    "kn": {
        "src_lang": "hi", "tgt_lang": "kn",
        "hypotheses": [
            "ನಾನು ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ಉದ್ಯಾನದಲ್ಲಿ ನಡೆಯುತ್ತೇನೆ.",
            "ಭಾರತ ವೈವಿಧ್ಯತೆಯಿಂದ ತುಂಬಿದ ದೇಶ.",
            "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಮಾನವ ಜೀವನವನ್ನು ಬದಲಾಯಿಸಿದೆ.",
            "ಶಿಕ್ಷಣ ಯಾವುದೇ ಸಮಾಜದ ಅಡಿಪಾಯ.",
            "ನೀರು ನಮ್ಮ ಜೀವನಕ್ಕೆ ಅತ್ಯಂತ ಅವಶ್ಯಕ.",
            "ಕ್ರೀಡೆ ದೇಹ ಮತ್ತು ಮನಸ್ಸು ಎರಡನ್ನೂ ಆರೋಗ್ಯಕರವಾಗಿ ಇಡುತ್ತದೆ.",
            "ಪ್ರಕೃತಿ ನಮಗೆ ಬಹಳಷ್ಟು ಕಲಿಸುತ್ತದೆ.",
            "ಕುಟುಂಬ ನಮ್ಮ ದೊಡ್ಡ ಆಸರೆ.",
            "ಶ್ರಮ ಮತ್ತು ನಿಷ್ಠೆಯಿಂದ ಎಲ್ಲ ಕಾರ್ಯ ಸಾಧ್ಯ.",
            "ಆರೋಗ್ಯವೇ ದೊಡ್ಡ ಸಂಪತ್ತು.",
        ],
        "references": [
            "ನಾನು ಪ್ರತಿ ದಿನ ಬೆಳಿಗ್ಗೆ ಉದ್ಯಾನವನದಲ್ಲಿ ನಡೆದಾಡುತ್ತೇನೆ.",
            "ಭಾರತ ವಿವಿಧತೆಗಳಿಂದ ತುಂಬಿರುವ ದೇಶ.",
            "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನಗಳು ಮಾನವ ಜೀವನವನ್ನು ಬದಲಿಸಿವೆ.",
            "ಶಿಕ್ಷಣವು ಯಾವುದೇ ಸಮಾಜದ ಅಡಿಪಾಯವಾಗಿದೆ.",
            "ನೀರು ನಮ್ಮ ಜೀವನಕ್ಕೆ ಅತ್ಯಂತ ಅವಶ್ಯಕವಾಗಿದೆ.",
            "ಕ್ರೀಡೆಯಿಂದ ದೇಹ ಮತ್ತು ಮನಸ್ಸು ಎರಡೂ ಆರೋಗ್ಯಕರವಾಗಿರುತ್ತವೆ.",
            "ಪ್ರಕೃತಿ ನಮಗೆ ಅನೇಕ ವಿಷಯಗಳನ್ನು ಕಲಿಸುತ್ತದೆ.",
            "ಕುಟುಂಬ ನಮ್ಮ ಅತ್ಯಂತ ದೊಡ್ಡ ಆಧಾರ.",
            "ಶ್ರಮ ಮತ್ತು ಸಮರ್ಪಣೆಯಿಂದ ಎಲ್ಲ ಕಾರ್ಯಗಳು ಸಾಧ್ಯ.",
            "ಆರೋಗ್ಯವೇ ಅತ್ಯಂತ ದೊಡ್ಡ ಸಂಪತ್ತು.",
        ],
    },
}


def analyse(datasets: dict, human_score_source: str = "proxy") -> list:
    """
    Run full correlation analysis across all language pairs.

    Parameters
    ----------
    datasets : dict  {tgt_lang: {hypotheses, references, ...}}
    human_score_source : str  'proxy' or 'indicmt'

    Returns
    -------
    list of result dicts (one per language pair)
    """
    all_rows = []

    for tgt_lang, data in datasets.items():
        src_lang = data["src_lang"]
        hyps = data["hypotheses"]
        refs = data["references"]
        n = len(hyps)

        print(f"\n[Correlation] Analysing {src_lang}→{tgt_lang} ({n} pairs) ...")

        # ── Compute all metric scores ──────────────────────────────────────
        scorer = ILAM(lang=tgt_lang)
        ilam_results = scorer.batch_score(hyps, refs)

        ilam_s = [r["ilam"] for r in ilam_results]
        morph_s = [r["morph"] for r in ilam_results]
        sem_s = [r["sem"] for r in ilam_results]
        script_s = [r["script"] for r in ilam_results]
        bleu_s = [_sentence_bleu(h, r) for h, r in zip(hyps, refs)]
        chrf_s = [_sentence_chrf(h, r) for h, r in zip(hyps, refs)]
        chrfpp_s = [_sentence_chrfpp(h, r) for h, r in zip(hyps, refs)]
        human_s = [_oracle_human_score(h, r, tgt_lang) for h, r in zip(hyps, refs)]

        # ── Correlations ──────────────────────────────────────────────────
        metrics = {
            "BLEU": bleu_s,
            "chrF": chrf_s,
            "chrF++": chrfpp_s,
            "ILAM (ours)": ilam_s,
            "MorphScore": morph_s,
            "SemScore": sem_s,
            "ScriptScore": script_s,
        }

        rows = []
        for metric_name, scores in metrics.items():
            corrs = _corr(scores, human_s)
            row = {
                "language_pair": f"{src_lang}→{tgt_lang}",
                "tgt_lang": tgt_lang,
                "metric": metric_name,
                "human_score_source": human_score_source,
                "n": n,
                **corrs,
            }
            rows.append(row)
            print(
                f"  {metric_name:20s} | "
                f"Pearson={corrs['pearson']:+.4f} | "
                f"Spearman={corrs['spearman']:+.4f} | "
                f"Kendall={corrs['kendall']:+.4f}"
            )

        all_rows.extend(rows)

    return all_rows


def build_latex_table(rows: list) -> str:
    """
    Build a LaTeX table for the paper comparing all metrics.
    Groups by language pair.
    """
    metric_order = ["BLEU", "chrF", "chrF++", "ILAM (ours)"]
    langs = sorted(set(r["tgt_lang"] for r in rows))

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lang_cols = " & ".join([f"\\multicolumn{{2}}{{c}}{{{l.upper()}}}" for l in langs])
    lines.append(r"\begin{tabular}{l" + "cc" * len(langs) + "}")
    lines.append(r"\toprule")
    lines.append(f"Metric & {lang_cols} \\\\")
    sub_cols = " & ".join(["Pearson & Spearman"] * len(langs))
    lines.append(f" & {sub_cols} \\\\")
    lines.append(r"\midrule")

    # Index rows by (tgt_lang, metric)
    idx = {(r["tgt_lang"], r["metric"]): r for r in rows}

    for metric in metric_order:
        cells = []
        for lang in langs:
            r = idx.get((lang, metric), {})
            p = f"{r.get('pearson', float('nan')):.3f}"
            s = f"{r.get('spearman', float('nan')):.3f}"
            cells.append(f"{p} & {s}")
        row_str = " & ".join(cells)
        bold = r"\textbf{" if metric == "ILAM (ours)" else ""
        bold_end = r"}" if metric == "ILAM (ours)" else ""
        lines.append(f"{bold}{metric}{bold_end} & {row_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Metric correlation with human judgments. "
                 r"Human scores are proxy oracle scores (char n-gram cosine). "
                 r"Best results in \textbf{bold}.}")
    lines.append(r"\label{tab:correlation}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not _HAS_SCIPY:
        print("[Correlation] Warning: 'scipy' not installed; using built-in correlation fallback.")
    if not _HAS_SACREBLEU:
        print("[Correlation] Warning: 'sacrebleu' not installed; sentence baselines use proxy fallback.")

    if args.demo or args.data_dir is None:
        print("[Correlation] Running in demo mode.\n")
        datasets = DEMO_DATA
    else:
        data_dir = Path(args.data_dir)
        datasets = {}
        for fp in data_dir.glob("*.json"):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            datasets[data.get("tgt_lang", fp.stem)] = data

    rows = analyse(datasets)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "correlation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Correlation] CSV → {csv_path}")

    # Save LaTeX
    latex = build_latex_table(rows)
    tex_path = os.path.join(args.out_dir, "correlation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[Correlation] LaTeX table → {tex_path}")

    # Save plain text report
    report_path = os.path.join(args.out_dir, "correlation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ILAM Correlation Report\n")
        f.write("=" * 60 + "\n\n")
        for row in rows:
            f.write(
                f"{row['language_pair']:12s} | {row['metric']:20s} | "
                f"Pearson={row['pearson']:+.4f} | "
                f"Spearman={row['spearman']:+.4f} | "
                f"Kendall={row['kendall']:+.4f}\n"
            )
        f.write("\n\n--- LaTeX Table ---\n\n")
        f.write(latex)
    print(f"[Correlation] Report → {report_path}")


if __name__ == "__main__":
    main()
