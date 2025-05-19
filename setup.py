from setuptools import setup, find_packages

setup(
    name="noisy-channel-correction",
    version="0.1.0",
    packages=find_packages(include=["lib", "lib.*"]),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.20.0",
        "python-Levenshtein>=0.12.2",
        "nltk>=3.6.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "streamlit>=1.0.0",
        "datasets>=2.0.0",
        "huggingface_hub>=0.0.19",
    ],
)
