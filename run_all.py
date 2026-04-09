"""
run_all.py
----------
End-to-end pipeline runner for the ILAM project.

Steps:
  1. (Optional) Translate Flores-200 using IndicTrans2
  2. Run baseline metrics (BLEU, chrF, chrF++)
  3. Run ILAM scoring
  4. Run correlation analysis
  5. Print full summary table

Usage:
    # Demo mode (no GPU needed, uses built-in samples)
    python run_all.py --demo

    # Full pipeline (requires GPU + internet for model downloads)
    python run_all.py --translate --src_lang hi --tgt_langs mr kn --max_samples 200

    # Score existing translation JSONs without re-running translation
    python run_all.py --translate --skip_translate_step
"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def print_banner(text: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def step_translate(args):
    print_banner("STEP 1: Cross-Lingual Transfer (IndicTrans2)")
    from transfer.translate import IndicTranslator
    translator = IndicTranslator(
        model_name="ai4bharat/indictrans2-indic-indic-1B",
        quantize=args.quantize,
    )
    allow_fallback = args.allow_builtin_fallback or (not args.strict_flores)
    results = translator.translate_flores200(
        src_lang=args.src_lang,
        tgt_langs=args.tgt_langs,
        split="devtest",
        max_samples=args.max_samples,
        save_dir="data/translations",
        allow_builtin_fallback=allow_fallback,
    )
    print(f"[run_all] Translation complete. Files saved to data/translations/")
    return results


def step_baselines(out_dir: str, demo: bool = True, data_dir: str = None, data_files: list = None):
    print_banner("STEP 2: Baseline Metrics (BLEU, chrF, chrF++)")
    from experiments.run_baselines import run_demo, score_file
    import csv

    os.makedirs(out_dir, exist_ok=True)

    if demo or data_dir is None:
        results = run_demo()
    else:
        files = data_files if data_files else sorted(Path(data_dir).glob("*.json"))
        results = [score_file(str(f)) for f in files]
        results = [r for r in results if r]

    out_path = str(Path(out_dir) / "baseline_scores.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"[run_all] Baselines saved → {out_path}")
    return results


def step_ilam(out_dir: str, demo: bool = True, data_dir: str = None, data_files: list = None):
    print_banner("STEP 3: ILAM Scoring")
    from experiments.run_ilam import DEMO_DATA, score_dataset, save_sentence_csv, compute_summary

    os.makedirs(out_dir, exist_ok=True)

    if demo or data_dir is None:
        datasets = DEMO_DATA
    else:
        datasets = {}
        files = data_files if data_files else sorted(Path(data_dir).glob("*.json"))
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            datasets[data.get("tgt_lang", fp.stem)] = data

    all_results = {}
    for tgt_lang, data in datasets.items():
        results = score_dataset(data, verbose=True)
        all_results[tgt_lang] = results
        save_sentence_csv(results, str(Path(out_dir) / f"ilam_scores_{tgt_lang}.csv"))

    summary = compute_summary(all_results)
    return all_results, summary


def step_correlation(out_dir: str, demo: bool = True, data_dir: str = None, data_files: list = None):
    print_banner("STEP 4: Correlation Analysis (Proxy Oracle)")
    from experiments.correlation import DEMO_DATA, analyse, build_latex_table
    import csv

    os.makedirs(out_dir, exist_ok=True)

    if demo or data_dir is None:
        datasets = DEMO_DATA
    else:
        datasets = {}
        files = data_files if data_files else sorted(Path(data_dir).glob("*.json"))
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            datasets[data.get("tgt_lang", fp.stem)] = data

    rows = analyse(datasets)

    with open(str(Path(out_dir) / "correlation_table.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    latex = build_latex_table(rows)
    with open(str(Path(out_dir) / "correlation_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex)

    with open(str(Path(out_dir) / "correlation_report.txt"), "w", encoding="utf-8") as f:
        f.write("ILAM Correlation Report (proxy human score source)\n")
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

    return rows


def print_final_summary(out_dir: str, baseline_results, ilam_summary, correlation_rows):
    print_banner("FINAL SUMMARY")

    print("\n── Baseline Metrics ──────────────────────────────────────────")
    for r in baseline_results:
        print(f"  {r.get('src_lang','?')}→{r.get('tgt_lang','?'):5s} | "
              f"BLEU={r.get('bleu', 0):6.2f} | "
              f"chrF={r.get('chrf', 0):6.2f} | "
              f"chrF++={r.get('chrfpp', 0):6.2f}")

    print("\n── ILAM Corpus Scores ────────────────────────────────────────")
    for r in ilam_summary:
        unicode_part = f" | Unicode={r['unicode']:.4f}" if "unicode" in r else ""
        print(f"  →{r['tgt_lang']:5s} | "
              f"ILAM={r['ilam']:.4f} | "
              f"Morph={r['morph']:.4f} | "
              f"Sem={r['sem']:.4f} | "
              f"Script={r['script']:.4f}{unicode_part}")

    print("\n── Correlation with Human Judgments (Pearson) ────────────────")
    for lang in sorted(set(r["tgt_lang"] for r in correlation_rows)):
        lang_rows = [r for r in correlation_rows if r["tgt_lang"] == lang]
        print(f"\n  Language: {lang}")
        for r in lang_rows:
            flag = " ◀ ILAM" if r["metric"] == "ILAM (ours)" else ""
            print(f"    {r['metric']:22s} Pearson={r['pearson']:+.4f} | "
                  f"Spearman={r['spearman']:+.4f}{flag}")

    print("\n── Output Files ──────────────────────────────────────────────")
    for f in sorted(Path(out_dir).glob("*")):
        print(f"  {f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ILAM End-to-End Pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Run with built-in sample data (no GPU needed)")
    parser.add_argument("--translate", action="store_true",
                        help="Run IndicTrans2 translation step (requires GPU)")
    parser.add_argument(
        "--skip_translate_step",
        action="store_true",
        help="When --translate is set, skip the translation step and score existing JSONs in data/translations.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for result files. Defaults: 'results' for demo, "
             "'results_flores' when --translate is used.",
    )
    parser.add_argument("--src_lang", type=str, default="hi")
    parser.add_argument("--tgt_langs", nargs="+", default=["mr", "kn"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--quantize", action="store_true",
                        help="Use 8-bit quantization for IndicTrans2")
    parser.add_argument(
        "--allow_builtin_fallback",
        action="store_true",
        help="If Flores-200 loading fails, use built-in 20-sentence sample data",
    )
    parser.add_argument(
        "--strict-flores",
        dest="strict_flores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), fail fast if Flores-200 loading fails. "
             "Disable with `--no-strict-flores` if you want to allow fallback behavior.",
    )
    args = parser.parse_args()

    if args.strict_flores and args.allow_builtin_fallback:
        parser.error("Conflicting flags: --strict-flores and --allow_builtin_fallback. Choose one.")

    os.makedirs("data/translations", exist_ok=True)

    demo = args.demo or not args.translate
    out_dir = args.out_dir or ("results" if demo else "results_flores")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = "data/translations" if args.translate else None
    selected_files = None

    if args.translate:
        selected_files = [
            Path(data_dir) / f"{args.src_lang}_{t}.json"
            for t in args.tgt_langs
        ]

    # Step 1: Translation (optional)
    if args.translate and not args.skip_translate_step:
        step_translate(args)

    # Step 2: Baselines
    baseline_results = step_baselines(out_dir=out_dir, demo=demo, data_dir=data_dir, data_files=selected_files)

    # Step 3: ILAM
    ilam_results, ilam_summary = step_ilam(out_dir=out_dir, demo=demo, data_dir=data_dir, data_files=selected_files)

    # Step 4: Correlation
    correlation_rows = step_correlation(out_dir=out_dir, demo=demo, data_dir=data_dir, data_files=selected_files)

    # Final summary
    print_final_summary(out_dir, baseline_results, ilam_summary, correlation_rows)


if __name__ == "__main__":
    main()
