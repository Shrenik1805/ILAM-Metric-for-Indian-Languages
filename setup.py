from setuptools import setup, find_packages

setup(
    name="ilam",
    version="0.1.0",
    description="Indian Language-Aware Metric for Indic NLP evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ILAM Research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "indic-nlp-library>=0.91",
        "sacrebleu>=2.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99",
        ],
        "transfer": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99",
            "accelerate>=0.24.0",
            "datasets>=2.14.0,<4.0.0",
            "IndicTransToolkit>=1.1.1",
        ],
        "finetune": [
            "peft>=0.6.0",
            "bitsandbytes>=0.41",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="nlp indic hindi marathi kannada evaluation metric machine-translation",
)
