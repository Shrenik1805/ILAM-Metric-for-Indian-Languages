"""
run_ilam.py
-----------
Run ILAM scoring on Hindi→Marathi and Hindi→Kannada transfer outputs.

Produces:
  - results/ilam_scores_mr.csv  — sentence-level scores for Marathi
  - results/ilam_scores_kn.csv  — sentence-level scores for Kannada
  - results/ilam_summary.csv    — corpus-level summary across all languages

Usage:
    python experiments/run_ilam.py --demo
    python experiments/run_ilam.py --data_dir data/translations
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ilam import ILAM


# ── Built-in demo data ────────────────────────────────────────────────────────

DEMO_DATA = {
    "mr": {
        "src_lang": "hi",
        "tgt_lang": "mr",
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
        "src_lang": "hi",
        "tgt_lang": "kn",
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


def score_dataset(data: dict, verbose: bool = True) -> list:
    """
    Run ILAM on a translation dataset dict.
    Returns list of per-sentence score dicts.
    """
    tgt_lang = data["tgt_lang"]
    hyps = data["hypotheses"]
    refs = data["references"]

    scorer = ILAM(lang=tgt_lang, verbose=verbose)
    results = scorer.batch_score(hyps, refs)

    # Add sentence index and source/target lang
    for i, r in enumerate(results):
        r["idx"] = i
        r["src_lang"] = data["src_lang"]
        r["tgt_lang"] = tgt_lang
        r["hypothesis"] = hyps[i]
        r["reference"] = refs[i]

    return results


def save_sentence_csv(results: list, out_path: str):
    """Save sentence-level results to CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["idx", "src_lang", "tgt_lang", "ilam", "morph", "sem", "script",
              "hypothesis", "reference"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"[ILAM] Sentence-level results → {out_path}")


def compute_summary(all_results: dict) -> list:
    """Compute corpus-level averages per language pair."""
    summary = []
    for tgt_lang, results in all_results.items():
        n = len(results)
        avg = {k: round(sum(r[k] for r in results) / n, 4)
               for k in ["ilam", "morph", "sem", "script"]}
        avg["tgt_lang"] = tgt_lang
        avg["n_sentences"] = n
        summary.append(avg)
        print(
            f"[ILAM Summary] {results[0]['src_lang']}→{tgt_lang} | "
            f"ILAM={avg['ilam']} | MorphScore={avg['morph']} | "
            f"SemScore={avg['sem']} | ScriptScore={avg['script']}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ILAM scoring")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable detailed ILAM progress logs (default: enabled).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.demo or args.data_dir is None:
        print("[ILAM] Running in demo mode with built-in sample data.\n")
        datasets = DEMO_DATA
    else:
        data_dir = Path(args.data_dir)
        datasets = {}
        for fp in sorted(data_dir.glob("*.json")):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            tgt = data.get("tgt_lang", fp.stem)
            datasets[tgt] = data

    all_results = {}
    for tgt_lang, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"[ILAM] Scoring {data['src_lang']}→{tgt_lang}")
        print(f"{'='*60}")
        results = score_dataset(data, verbose=args.verbose)
        all_results[tgt_lang] = results
        out_path = os.path.join(args.out_dir, f"ilam_scores_{tgt_lang}.csv")
        save_sentence_csv(results, out_path)

    if not all_results:
        print("[ILAM] No datasets found to score.")
        return

    print(f"\n{'='*60}")
    print("[ILAM] Corpus-level summary")
    print(f"{'='*60}")
    summary = compute_summary(all_results)

    summary_path = os.path.join(args.out_dir, "ilam_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = ["tgt_lang", "n_sentences", "ilam", "morph", "sem", "script"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[ILAM] Summary → {summary_path}")


if __name__ == "__main__":
    main()
