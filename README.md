# AAC Noisy Input Correction Engine

A lightweight, interpretable correction system for noisy AAC (Augmentative and Alternative Communication) input. This system uses a **Noisy Channel Model** framework to correct user-generated text errors commonly seen in switch-access, eyegaze, dwell typing, and other AAC modalities.

It leverages a high-performance **PPM language model** to score intended inputs (`P(intended)`) and a configurable **error model** to estimate `P(noisy | intended)` based on simulated input errors.

---

## 💡 Use Case

AAC users often make input errors due to physical or interface constraints. This project helps correct such errors in real time or during post-processing, improving intelligibility and user experience without requiring a neural network.

---

## 🔧 Architecture

The system is modular and easy to customize or expand:

```
noisy-channel-correction/
├── lib/                      # Shared library code
│   ├── noise_model/          # Noise simulation
│   ├── confusion_matrix/     # Confusion matrix generation
│   ├── candidate_generator/  # Candidate generation
│   ├── corrector/            # Correction engine
│   ├── pylm/                 # Language model code
│   └── models/               # Model files
├── data/                     # Data files
│   ├── conversational_corpora/  # Training corpus data
│   ├── evaluation_results/      # Evaluation results
│   └── visualizations/          # Visualization outputs
├── scripts/                  # Utility scripts
│   ├── noise_simulator/         # Noise simulator scripts
│   ├── confusion_matrix_builder/# Confusion matrix builder scripts
│   ├── candidate_generator/     # Candidate generator scripts
│   ├── correction_engine/       # Correction engine scripts
│   └── evaluation/              # Evaluation scripts
├── tests/                    # Test files
└── demo/                     # Demo applications
    ├── noise_simulator/         # Noise simulator demos
    ├── confusion_matrix_builder/# Confusion matrix builder demos
    ├── candidate_generator/     # Candidate generator demos
    ├── correction_engine/       # Correction engine demos
    └── evaluation/              # Evaluation demos
```

Each module is standalone and reusable in other AAC or correction projects.

---

## 📦 Features

- **Modular Error Simulation**: Simulate access-method-specific errors (missed hits, double hits, proximity errors).
- **Character-Level Confusion Matrix**: Tuned from realistic simulated noise, supports insertion, deletion, substitution, and transposition.
- **Flexible Candidate Generation**: Efficiently generate and filter plausible corrections using edit distance, keyboard adjacency, and lexicon constraints.
- **Probabilistic Correction Engine**: Ranks candidates using:

score = log P(intended) + log P(noisy | intended)

Powered by your enhanced PPM model.

- **No Neural Dependencies**: Designed for low-resource, interpretable deployment.

---

## 🛠 Setup

Requirements:
- Python 3.8+
- Dependencies listed in `requirements.txt` (e.g. Levenshtein, numpy)

```bash
git clone https://github.com/your-org/aac-corrector.git
cd aac-corrector
pip install -r requirements.txt


⸻

## 🚀 Usage

1. Simulate Errors

```bash
python scripts/noise_simulator/simulate.py --input data/wordlist.txt --output data/noisy_pairs.json
```

2. Build Confusion Matrix

```bash
python scripts/confusion_matrix_builder/build_matrix.py --input data/noisy_pairs.json --output data/confusion_matrix.json
```

3. Generate Candidates

```bash
python scripts/candidate_generator/generate_candidates.py --input "thes is a tst" --lexicon data/comprehensive_lexicon.txt
```

4. Run Correction

```bash
python scripts/correction_engine/correct.py --input "thes is a tst" --output corrected.txt
```

5. Run Demo

```bash
python scripts/evaluation/demo.py
```


⸻

🧪 Evaluation

Use demo/eval.py to test accuracy of correction across datasets with varying noise profiles. Includes:
	•	Accuracy@1
	•	Accuracy@N
	•	Context-aware reranking (optional)

⸻

📚 Future Enhancements
	•	Access-specific confusion models (eyegaze vs touch vs switch)
	•	Word-level confusion matrices
	•	Beam search decoding
	•	Integration with AAC systems or TTS pipelines

⸻

👥 Credits

Developed by [Will Wade
Inspired by AAC research and real-world interaction data
PPM model by [willwade (pylm repo and in turn the jslm repo from google/Brian Roark)]

