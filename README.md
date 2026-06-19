# HackerRank Orchestrate (June 2026) - Multi-Modal Evidence Review

This repository contains an automated verification system for multimodal insurance damage claims.

## Setup Instructions

### 1. Get a Gemini API Key
The pipeline is powered by `gemini-2.5-flash` using the `google-genai` SDK.
You can get a free API key (no credit card required) at [Google AI Studio](https://aistudio.google.com/).

### 2. Configure Environment
Create a `.env` file in the root of the repository and add your API key:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Generating Final Predictions
To run the system against the unlabeled test set (`dataset/claims.csv`):
```bash
python code/main.py
```
This will produce `output.csv` in the repository root, formatted exactly as required by the problem statement.

### Running Evaluation
To run the system against the labeled sample set and generate an evaluation report:
```bash
python code/evaluation/main.py
```
This script will:
1. Run the pipeline against `dataset/sample_claims.csv`.
2. Compare the generated predictions against the ground truth labels.
3. Compute accuracy and exact-match rates.
4. Output a detailed analysis to `evaluation/evaluation_report.md`.
