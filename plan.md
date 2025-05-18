
PROJECT PLAN: Noisy Channel Correction System for AAC Input

Goal:

Correct noisy AAC input (word or character sequences) using a probabilistic model that ranks candidate corrections by combining:
	•	Your existing PPM language model as the prior (P(intended))
	•	A custom error model for P(noisy | intended) based on simulated AAC-style input errors

⸻

MODULE 1: Data + Noise Simulator

Objectives:
	•	Build a clean word or phrase corpus (from AAC datasets or wordlists)
	•	Create simulated noisy versions of each word/phrase using:
	•	Keyboard layout confusion (QWERTY/ABC)
	•	Dwell-time errors: missing or doubled letters
	•	Transpositions
	•	Access-specific biases (optional later)

Deliverables:
	•	noise_model.py with configurable error generators
	•	A training set of (intended, noisy) pairs for confusion matrix generation

⸻

MODULE 2: Confusion Matrix Builder

Objectives:
	•	From the simulated (intended, noisy) pairs, count error types:
	•	Substitutions, deletions, insertions, transpositions
	•	Normalize to get P(noisy_char | intended_char)
	•	Optionally extend to bigram or word-level if needed

Deliverables:
	•	confusion_matrix.py
	•	Function: build_confusion_matrix(pairs)
	•	Function: get_error_probability(noisy, intended)

⸻

MODULE 3: Candidate Generator

Objectives:
	•	For a given noisy input, generate plausible candidates
	•	Edit distance 1–2 (with early pruning)
	•	Lexicon- or corpus-based filtering
	•	Optional: restrict to completions from previous word

Deliverables:
	•	candidate_generator.py
	•	Function: generate_candidates(noisy_input, lexicon)
	•	Integrates edit-distance + adjacent-key swaps + common misspellings

⸻

MODULE 4: Correction Engine (Noisy Channel Model)

Objectives:
	•	Use your PPM model to compute P(intended)
	•	Use error model to compute P(noisy | intended)
	•	Rank candidates by:

score = log(P(intended)) + log(P(noisy | intended))


	•	Return top-N corrected candidates

Deliverables:
	•	corrector.py
	•	Class: NoisyChannelCorrector(ppm_model, confusion_model)
	•	Method: correct(noisy_input) -> corrected_output

⸻

MODULE 5: Interface and Evaluation

Objectives:
	•	Build an interactive test UI or CLI
	•	Load sample AAC-like utterances and visualize corrections
	•	Evaluate:
	•	Accuracy@1 and Accuracy@N
	•	Confidence thresholds
	•	Speed of inference

Deliverables:
	•	demo.py (CLI or Streamlit)
	•	eval.py with accuracy, precision, recall
	•	Test dataset (could include simulated logs)

⸻

Stretch Goals
	•	Context-aware correction: use PPM with multi-token context
	•	Beam search for longer utterance corrections
	•	Dynamic access model switching based on user profile

