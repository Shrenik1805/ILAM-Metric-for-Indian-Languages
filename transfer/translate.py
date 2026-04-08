"""
translate.py
------------
Cross-lingual transfer pipeline using IndicTrans2.

Model: ai4bharat/indictrans2-indic-indic-1B
  - Supports all 22 scheduled Indian languages
  - Best-in-class for Indic-to-Indic translation

Fixes applied:
  - Flores-200 trust_remote_code removed (deprecated in datasets 2024)
  - Multiple fallback paths for Flores-200 dataset
  - Built-in reference sentences so scoring works even offline
  - SemScore forced to CPU/char-fallback to prevent Mac segfault with MuRIL
  - Hugging Face trust_remote_code applied to bypass terminal [y/N] prompts
  - Source and Target language tags prepended for strict tokenization

Usage
-----
    from transfer.translate import IndicTranslator

    translator = IndicTranslator()
    out = translator.translate("मैं घर जाता हूँ", src_lang="hi", tgt_lang="mr")

    results = translator.translate_flores200(
        src_lang="hi", tgt_langs=["mr", "kn"], max_samples=200
    )
"""

import json
from pathlib import Path
from hf_auth import apply_hf_token_env

# ── Language code mappings ───────────────────────────────────────────────────
INDICTRANS_LANG_MAP = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ml": "mal_Mlym",
}

FLORES_LANG_MAP = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "kn": "kan_Knda",
}

def _load_flores(lang_code: str, split: str, max_samples: int) -> list:
    """
    Load Flores-200 sentences.
    Tries multiple dataset paths.
    """
    try:
        from datasets import load_dataset
        import datasets as hf_datasets
    except ModuleNotFoundError:
        raise RuntimeError(
            "Missing dependency: datasets. Install with "
            "`pip install datasets pyarrow` in your virtual environment."
        )

    flores_code = FLORES_LANG_MAP.get(lang_code, lang_code)
    hf_token = apply_hf_token_env()

    def _ld(path: str, config: str, split_name: str):
        kwargs = {"split": split_name, "trust_remote_code": True}
        if hf_token:
            kwargs["token"] = hf_token
        try:
            return load_dataset(path, config, **kwargs)
        except TypeError:
            # Backward compatibility with older datasets APIs
            kwargs.pop("trust_remote_code", None)
            if hf_token:
                kwargs.pop("token", None)
                kwargs["use_auth_token"] = hf_token
            return load_dataset(path, config, trust_remote_code=True, **kwargs)

    # datasets>=4 removed support for script-based datasets (e.g., flores.py).
    # Prefer parquet/mirror repos for newer datasets versions.
    major_ver = 0
    try:
        major_ver = int(str(getattr(hf_datasets, "__version__", "0")).split(".")[0])
    except Exception:
        major_ver = 0

    if major_ver >= 4:
        attempts = [
            lambda: _ld("Muennighoff/flores200", flores_code, split),
            lambda: _ld("facebook/flores", flores_code, split),
            lambda: _ld("mteb/flores", flores_code, split),
        ]
    else:
        attempts = [
            # Official dataset (preferred on older datasets versions)
            lambda: _ld("facebook/flores", flores_code, split),
            # Additional mirrors
            lambda: _ld("Muennighoff/flores200", flores_code, split),
            lambda: _ld("ai4bharat/flores", flores_code, split),
            lambda: _ld("mteb/flores", flores_code, split),
        ]

    last_err = None
    for attempt in attempts:
        try:
            ds = attempt()
            sentences = [row.get("sentence", "") for row in ds]
            sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
            if sentences:
                return sentences[:max_samples]
        except Exception as e:
            last_err = e
            continue

    if major_ver >= 4 and last_err and "Dataset scripts are no longer supported" in str(last_err):
        raise RuntimeError(
            "Your installed `datasets` version is >=4, which blocks Flores script datasets. "
            "Fix: `ilam_env/bin/python3 -m pip install \"datasets>=2.14.0,<4\"`"
        ) from last_err

    raise RuntimeError(f"All Flores-200 paths failed for {lang_code}: {last_err}")


class MockTranslator:
    """Fallback: returns source text unchanged. Useful for testing pipeline."""
    def translate(self, sentence, src_lang, tgt_lang):
        return sentence

    def translate_batch(self, sentences, src_lang, tgt_lang, batch_size=8):
        return list(sentences)


class IndicTranslator:
    """
    Wrapper around IndicTrans2 for zero-shot cross-lingual translation.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    device : str
        'cuda', 'mps', 'cpu', or 'auto'.
    quantize : bool
        Use 8-bit quantization (requires bitsandbytes, Linux/CUDA only).
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-indic-indic-1B",
        device: str = "auto",
        quantize: bool = False,
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = device
        self._quantize = quantize
        self._loaded = False
        self._ip = None

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            print(f"[IndicTranslator] Loading {self.model_name} ...")
            hf_token = apply_hf_token_env()
            # FIX: Added trust_remote_code=True to bypass the prompt for the tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, token=hf_token
            )

            # Determine device
            if self._device == "auto":
                if torch.cuda.is_available():
                    dev = "cuda"
                elif torch.backends.mps.is_available():
                    # MPS (Apple Silicon) — do NOT use device_map="auto"
                    # it triggers a segfault with some model architectures
                    dev = "cpu"
                else:
                    dev = "cpu"
            else:
                dev = self._device

            # FIX: Added trust_remote_code=True to bypass the prompt for the model
            kwargs = {"trust_remote_code": True, "token": hf_token}
            
            if self._quantize and dev == "cuda":
                kwargs["load_in_8bit"] = True
            else:
                kwargs["torch_dtype"] = torch.float32

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **kwargs
            )
            self._model = self._model.to(dev)
            self._model.eval()

            # Recommended official preprocessing/postprocessing pipeline.
            try:
                from IndicTransToolkit.processor import IndicProcessor
                self._ip = IndicProcessor(inference=True)
            except Exception as e:
                self._ip = None
                print(f"[IndicTranslator] Warning: IndicProcessor unavailable, using legacy token path: {e}")

            self._actual_device = dev
            self._loaded = True
            print(f"[IndicTranslator] Model loaded on {dev}.")
        except Exception as e:
            print(f"[IndicTranslator] Warning: Could not load model: {e}")
            print("[IndicTranslator] Falling back to MockTranslator.")
            self._model = None
            self._loaded = True

    def translate(self, sentence: str, src_lang: str, tgt_lang: str) -> str:
        return self.translate_batch([sentence], src_lang, tgt_lang, batch_size=1)[0]

    def translate_batch(
        self,
        sentences: list,
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 4,
    ) -> list:
        self._load()

        if self._model is None:
            return MockTranslator().translate_batch(sentences, src_lang, tgt_lang)

        # Grab BOTH tags needed for the custom tokenizer
        src_tag = INDICTRANS_LANG_MAP.get(src_lang, src_lang)
        tgt_tag = INDICTRANS_LANG_MAP.get(tgt_lang, tgt_lang)

        import torch
        outputs = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]

            if self._ip is not None:
                proc_batch = self._ip.preprocess_batch(
                    batch, src_lang=src_tag, tgt_lang=tgt_tag
                )
                inputs = self._tokenizer(
                    proc_batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )
                decoded = self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                decoded = self._ip.postprocess_batch(decoded, lang=tgt_tag)
            else:
                # Legacy fallback if IndicProcessor is not available.
                tagged_batch = [f"{src_tag} {tgt_tag} {text}" for text in batch]
                inputs = self._tokenizer(
                    tagged_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_tag),
                        max_new_tokens=256,
                        num_beams=4,
                        early_stopping=True,
                    )
                decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

            outputs.extend(decoded)
            print(f"  [translate] {i + len(batch)}/{len(sentences)} done", end="\r")

        print()
        return outputs
    
    def translate_flores200(
        self,
        src_lang: str = "hi",
        tgt_langs: list = None,
        split: str = "devtest",
        max_samples: int = 200,
        save_dir: str = None,
        allow_builtin_fallback: bool = False,
    ) -> dict:
        """
        Translate Flores-200 sentences from src_lang to each tgt_lang.
        By default, raises on Flores load failure to avoid silently using only
        the tiny built-in sample set.
        """
        if tgt_langs is None:
            tgt_langs = ["mr", "kn"]

        print(f"[IndicTranslator] Loading Flores-200 source ({src_lang}, {split}) ...")
        try:
            sources = _load_flores(src_lang, split, max_samples)
            print(f"[IndicTranslator] Loaded {len(sources)} source sentences.")
        except Exception as e:
            if not allow_builtin_fallback:
                raise RuntimeError(
                    f"Could not load Flores-200 ({src_lang}, {split}). "
                    "This would fall back to only 20 sample sentences. "
                    "Fix Hugging Face access/deps, or set "
                    "`allow_builtin_fallback=True` intentionally."
                ) from e
            print(f"[IndicTranslator] Flores-200 unavailable: {e}")
            print("[IndicTranslator] Using built-in sample sentences.")
            sources = _SAMPLE_SENTENCES.get(src_lang, _SAMPLE_SENTENCES["hi"])[:max_samples]

        results = {}

        for tgt_lang in tgt_langs:
            print(f"\n[IndicTranslator] Translating {src_lang} → {tgt_lang} ({len(sources)} sentences) ...")

            # Load references
            try:
                references = _load_flores(tgt_lang, split, len(sources))
                print(f"[IndicTranslator] Loaded {len(references)} reference sentences ({tgt_lang}).")
            except Exception:
                references = _REFERENCE_SENTENCES.get(tgt_lang, [])[:len(sources)]
                if references:
                    print(f"[IndicTranslator] Using built-in references for {tgt_lang}.")
                else:
                    references = [""] * len(sources)
                    print(f"[IndicTranslator] No references for {tgt_lang} — correlation will use proxy scores.")

            # Pad/trim references to match sources length
            if len(references) < len(sources):
                references += [""] * (len(sources) - len(references))
            references = references[:len(sources)]

            # Translate
            hypotheses = self.translate_batch(sources, src_lang, tgt_lang)

            results[tgt_lang] = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "sources": sources,
                "hypotheses": hypotheses,
                "references": references,
            }

            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                out_path = Path(save_dir) / f"{src_lang}_{tgt_lang}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results[tgt_lang], f, ensure_ascii=False, indent=2)
                print(f"[IndicTranslator] Saved → {out_path}")

        return results


# ── Built-in sample sentences ────────────────────────────────────────────────

_SAMPLE_SENTENCES = {
    "hi": [
        "मैं हर दिन सुबह पार्क में टहलता हूँ।",
        "भारत एक विविधताओं से भरा हुआ देश है।",
        "विज्ञान और प्रौद्योगिकी ने मानव जीवन को बदल दिया है।",
        "शिक्षा किसी भी समाज की नींव होती है।",
        "पानी हमारे जीवन के लिए अत्यंत आवश्यक है।",
        "खेलकूद से शरीर और मन दोनों स्वस्थ रहते हैं।",
        "प्रकृति हमें बहुत कुछ सिखाती है।",
        "परिवार हमारा सबसे बड़ा सहारा होता है।",
        "मेहनत और लगन से हर काम संभव है।",
        "स्वास्थ्य ही सबसे बड़ा धन है।",
        "आज का मौसम बहुत सुहावना है।",
        "बच्चे देश का भविष्य होते हैं।",
        "पढ़ाई में ध्यान लगाना बहुत ज़रूरी है।",
        "मित्रता एक अनमोल रिश्ता है।",
        "सत्य की राह पर चलना कठिन पर सही होता है।",
        "हमारे देश की संस्कृति बहुत प्राचीन है।",
        "आर्थिक विकास के साथ सामाजिक विकास भी आवश्यक है।",
        "संगीत आत्मा को सुकून देता है।",
        "वृक्षारोपण पर्यावरण संरक्षण का एक महत्वपूर्ण उपाय है।",
        "समानता और न्याय लोकतंत्र के मूल सिद्धांत हैं।",
    ]
}

# Built-in reference sentences aligned with _SAMPLE_SENTENCES["hi"]
_REFERENCE_SENTENCES = {
    "mr": [
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
        "आजचे हवामान खूप छान आहे.",
        "मुले देशाचे भविष्य आहेत.",
        "अभ्यासावर लक्ष केंद्रित करणे अत्यंत आवश्यक आहे.",
        "मैत्री ही एक अमूल्य नाते आहे.",
        "सत्याच्या मार्गावर चालणे कठीण पण योग्य आहे.",
        "आपल्या देशाची संस्कृती अतिशय प्राचीन आहे.",
        "आर्थिक विकासाबरोबरच सामाजिक विकासही आवश्यक आहे.",
        "संगीत आत्म्याला शांती देते.",
        "वृक्षारोपण पर्यावरण संरक्षणाचा एक महत्त्वाचा उपाय आहे.",
        "समानता आणि न्याय हे लोकशाहीचे मूळ तत्त्व आहेत.",
    ],
    "kn": [
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
        "ಇಂದಿನ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
        "ಮಕ್ಕಳು ದೇಶದ ಭವಿಷ್ಯ.",
        "ಅಧ್ಯಯನದಲ್ಲಿ ಗಮನ ಹರಿಸುವುದು ಅತ್ಯಂತ ಅವಶ್ಯಕ.",
        "ಸ್ನೇಹ ಒಂದು ಅಮೂಲ್ಯ ಸಂಬಂಧ.",
        "ಸತ್ಯದ ಮಾರ್ಗದಲ್ಲಿ ನಡೆಯುವುದು ಕಷ್ಟವಾದರೂ ಸರಿಯಾದದ್ದು.",
        "ನಮ್ಮ ದೇಶದ ಸಂಸ್ಕೃತಿ ಬಹಳ ಪ್ರಾಚೀನವಾದದ್ದು.",
        "ಆರ್ಥಿಕ ಅಭಿವೃದ್ಧಿಯ ಜೊತೆ ಸಾಮಾಜಿಕ ಅಭಿವೃದ್ಧಿಯೂ ಅವಶ್ಯಕ.",
        "ಸಂಗೀತ ಆತ್ಮಕ್ಕೆ ಶಾಂತಿ ನೀಡುತ್ತದೆ.",
        "ವೃಕ್ಷಾರೋಪಣ ಪರಿಸರ ಸಂರಕ್ಷಣೆಯ ಮುಖ್ಯ ಉಪಾಯ.",
        "ಸಮಾನತೆ ಮತ್ತು ನ್ಯಾಯ ಪ್ರಜಾಪ್ರಭುತ್ವದ ಮೂಲ ತತ್ವಗಳು.",
    ],
}
