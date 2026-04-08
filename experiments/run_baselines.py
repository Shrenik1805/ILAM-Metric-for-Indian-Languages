"""
run_baselines.py
----------------
Compute BLEU, chrF, chrF++ baseline scores for Hindi→Marathi and
Hindi→Kannada transfer outputs.

Outputs a CSV: results/baseline_scores.csv

Usage:
    python experiments/run_baselines.py --data_dir data/translations
    python experiments/run_baselines.py --demo   # runs on built-in samples
"""

import argparse
import json
import csv
import os
from pathlib import Path
from collections import Counter

try:
    import sacrebleu
    from sacrebleu.metrics import CHRF
    _HAS_SACREBLEU = True
except Exception:
    sacrebleu = None
    CHRF = None
    _HAS_SACREBLEU = False


def _token_f1_score(hypothesis: str, reference: str) -> float:
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    hyp_counts = Counter(hyp_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((hyp_counts & ref_counts).values())
    precision = overlap / max(1, len(hyp_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


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


def bleu_score(hypotheses: list, references: list) -> float:
    """Corpus-level BLEU."""
    if _HAS_SACREBLEU:
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return round(result.score, 2)

    # Fallback approximation when sacrebleu is unavailable.
    scores = [_token_f1_score(h, r) for h, r in zip(hypotheses, references)]
    return round((sum(scores) / max(1, len(scores))) * 100.0, 2)


def chrf_score(hypotheses: list, references: list, word_order: int = 0) -> float:
    """Corpus-level chrF (word_order=0) or chrF++ (word_order=2)."""
    if _HAS_SACREBLEU:
        metric = CHRF(word_order=word_order)
        result = metric.corpus_score(hypotheses, [references])
        return round(result.score, 2)

    # Fallback approximation when sacrebleu is unavailable.
    n = 6 if word_order == 0 else 4
    scores = [_char_f_score(h, r, n=n) for h, r in zip(hypotheses, references)]
    return round((sum(scores) / max(1, len(scores))) * 100.0, 2)


def score_file(filepath: str) -> dict:
    """
    Load a JSON translation file and compute all baseline metrics.

    Expected JSON format:
      {
        "src_lang": "hi",
        "tgt_lang": "mr",
        "sources": [...],
        "hypotheses": [...],
        "references": [...]
      }
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    hyps = data["hypotheses"]
    refs = data["references"]
    src_lang = data.get("src_lang", "?")
    tgt_lang = data.get("tgt_lang", "?")

    # Filter out empty references
    pairs = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
    if not pairs:
        print(f"[Baselines] Warning: No valid reference pairs in {filepath}")
        return {}

    hyps_f, refs_f = zip(*pairs)

    results = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_sentences": len(hyps_f),
        "bleu": bleu_score(list(hyps_f), list(refs_f)),
        "chrf": chrf_score(list(hyps_f), list(refs_f), word_order=0),
        "chrfpp": chrf_score(list(hyps_f), list(refs_f), word_order=2),
    }

    print(
        f"[Baselines] {src_lang}→{tgt_lang} | n={results['n_sentences']} | "
        f"BLEU={results['bleu']} | chrF={results['chrf']} | chrF++={results['chrfpp']}"
    )
    return results


def run_demo():
    """Run baseline metrics on built-in sample data."""
    # Sample Hindi→Marathi pairs (illustrative)
    demo_data = {
        "mr": {
            "src_lang": "hi",
            "tgt_lang": "mr",
            "hypotheses": [
                "मी रोज सकाळी उद्यानात फिरतो.",
                "भारत हा विविधतेने भरलेला देश आहे.",
                "विज्ञान आणि तंत्रज्ञानाने मानवी जीवन बदलले आहे.",
                "शिक्षण हा कोणत्याही समाजाचा पाया असतो.",
                "पाणी हे आपल्या जीवनासाठी अत्यंत आवश्यक आहे.",
            ],
            "references": [
                "मी दररोज सकाळी बागेत फेरफटका मारतो.",
                "भारत हा विविधतांनी भरलेला देश आहे.",
                "विज्ञान व तंत्रज्ञानाने माणसाचे जीवन बदलून टाकले आहे.",
                "शिक्षण हा कोणत्याही समाजाचा आधारस्तंभ असतो.",
                "पाणी हे आपल्या जीवनासाठी अत्यंत महत्त्वाचे आहे.",
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
            ],
            "references": [
                "ನಾನು ಪ್ರತಿ ದಿನ ಬೆಳಿಗ್ಗೆ ಉದ್ಯಾನವನದಲ್ಲಿ ನಡೆದಾಡುತ್ತೇನೆ.",
                "ಭಾರತ ವಿವಿಧತೆಗಳಿಂದ ತುಂಬಿರುವ ದೇಶ.",
                "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನಗಳು ಮಾನವ ಜೀವನವನ್ನು ಬದಲಿಸಿವೆ.",
                "ಶಿಕ್ಷಣವು ಯಾವುದೇ ಸಮಾಜದ ಅಡಿಪಾಯವಾಗಿದೆ.",
                "ನೀರು ನಮ್ಮ ಜೀವನಕ್ಕೆ ಅತ್ಯಂತ ಅವಶ್ಯಕವಾಗಿದೆ.",
            ],
        },
    }

    all_results = []
    for tgt_lang, data in demo_data.items():
        hyps = data["hypotheses"]
        refs = data["references"]
        src = data["src_lang"]
        tgt = data["tgt_lang"]
        result = {
            "src_lang": src,
            "tgt_lang": tgt,
            "n_sentences": len(hyps),
            "bleu": bleu_score(hyps, refs),
            "chrf": chrf_score(hyps, refs, 0),
            "chrfpp": chrf_score(hyps, refs, 2),
        }
        print(
            f"[Baselines] {src}→{tgt} | "
            f"BLEU={result['bleu']} | chrF={result['chrf']} | chrF++={result['chrfpp']}"
        )
        all_results.append(result)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run baseline NLP metrics")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing JSON translation files")
    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in sample data")
    parser.add_argument("--out", type=str, default="results/baseline_scores.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if not _HAS_SACREBLEU:
        print("[Baselines] Warning: 'sacrebleu' not installed; using fallback metric approximations.")

    if args.demo or args.data_dir is None:
        results = run_demo()
    else:
        data_dir = Path(args.data_dir)
        json_files = sorted(data_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {data_dir}")
            return
        results = [score_file(str(f)) for f in json_files]
        results = [r for r in results if r]

    # Write CSV
    if results:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Baselines] Results saved to {args.out}")


if __name__ == "__main__":
    main()
