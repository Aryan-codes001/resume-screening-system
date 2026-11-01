
# AI-Based Resume Screening System

An AI-powered resume screening prototype that compares candidate resumes
with a job description, calculates relevance scores, and ranks candidates
using semantic similarity and required-skill coverage.

## Project Objectives

- Read resumes in TXT, PDF, and DOCX formats.
- Preprocess text before comparison.
- Generate semantic embeddings using Sentence Transformers.
- Use TF-IDF as a fallback if the transformer model cannot load.
- Calculate cosine similarity between resumes and the job description.
- Identify matched and missing skills.
- Rank candidates and export results to CSV and PNG.

## Technology Stack

- Python
- Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn
- pandas
- Matplotlib
- pypdf
- python-docx
- Jupyter Notebook

## Project Structure

```text
ai-resume-screening-system/
├── data/
│   ├── job_description/
│   │   └── machine_learning_engineer_nlp.txt
│   └── resumes/
│       └── sample resume files
├── results/
│   ├── ranking.csv
│   └── ranking.png
├── screen.py
├── resume_screening.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

1. Load the job description and candidate resumes.
2. Extract text from TXT, PDF, or DOCX files.
3. Clean the text by removing contact details, punctuation, and stop words.
4. Generate text embeddings using Sentence Transformers.
5. Calculate cosine similarity between each resume and the job description.
6. Calculate the proportion of required skills detected in each resume.
7. Combine the scores and rank the candidates.
8. Save the results to CSV and generate a ranking chart.

## Scoring Method

The final score combines semantic similarity and skill coverage:

**Final Score = (0.60 × Scaled Semantic Similarity + 0.40 × Skill Coverage) × 100**

Semantic similarity is scaled relative to the resumes in the current batch.

The final score is a ranking aid, not a probability of getting hired.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-resume-screening-system
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use a different compatible Python version, replace `py -3.10` accordingly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the screening system

```bash
python screen.py
```

### 5. Run the Jupyter Notebook

Open `resume_screening.ipynb` in VS Code, select the project's Python environment, and run the cells in order.

## Output Files

### `results/ranking.csv`

Contains candidate ranks, semantic similarity, skill coverage, final scores, matched skills, missing skills, and extracted experience estimates.

### `results/ranking.png`

Visual comparison of candidate final scores.

## Sample Dataset

This project includes eight synthetic sample resumes representing different levels of relevance to a Machine Learning Engineer (NLP) job description.

The dataset is intended for demonstration and educational use.

## Limitations

- Skill matching uses predefined keyword and alias rules.
- Experience extraction is approximate and may return a missing value.
- Scanned PDFs may require OCR, which is not currently implemented.
- Semantic scores depend on the model and available resume text.
- Min-max scaling makes scores dependent on the resumes in the current batch.
- The system may miss relevant skills expressed using unfamiliar terminology.
- Results must not be used as the sole basis for hiring decisions.

## Future Improvements

- Build a web interface for uploading resumes.
- Add OCR support for scanned resumes.
- Improve skill extraction using NLP.
- Add configurable job-specific skill weights.
- Create evaluation tests using a manually reviewed dataset.
- Add fairness, privacy, and explainability checks.

## Disclaimer

This is an educational prototype. It does not make hiring decisions and should not replace qualified human review.
