
from pathlib import Path
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESUME_DIR = BASE_DIR / "data" / "resumes"
JOB_DESCRIPTION_PATH = (
    BASE_DIR / "data" / "job_description"
    / "machine_learning_engineer_nlp.txt"
)
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Final score = 60% semantic similarity + 40% skill coverage
SEMANTIC_WEIGHT = 0.60
SKILL_WEIGHT = 0.40

REQUIRED_SKILLS = [
    "Python",
    "Natural Language Processing",
    "Machine Learning",
    "Deep Learning",
    "Transformers",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "pandas",
    "NumPy",
    "Text Preprocessing",
    "Sentence Embeddings",
    "Model Evaluation",
    "Git",
]

SKILL_ALIASES = {
    "Natural Language Processing": ["NLP"],
    "Machine Learning": ["ML"],
    "Deep Learning": ["DL"],
    "Transformers": ["Transformer Models"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "NumPy": ["numpy"],
    "Sentence Embeddings": ["Sentence Embedding"],
    "Model Evaluation": ["Model Performance Evaluation"],
}


# --------------------------------------------------
# 2. EXTRACT TEXT FROM TXT, PDF AND DOCX
# --------------------------------------------------

def extract_text(file_path):
    """Extract readable text from a supported resume file."""

    extension = file_path.suffix.lower()

    try:
        if extension == ".txt":
            return file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        elif extension == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            pages = []

            for page in reader.pages:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)

        elif extension == ".docx":
            from docx import Document

            document = Document(str(file_path))
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
            ]

            # Also read text inside Word tables.
            for table in document.tables:
                for row in table.rows:
                    paragraphs.append(
                        " ".join(cell.text for cell in row.cells)
                    )

            return "\n".join(paragraphs)

        else:
            warnings.warn(
                f"Unsupported file type skipped: {file_path.name}"
            )
            return ""

    except Exception as error:
        warnings.warn(
            f"Could not read {file_path.name}: {error}"
        )
        return ""


# --------------------------------------------------
# 3. PREPROCESS TEXT
# --------------------------------------------------

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    """Remove contact details, URLs, punctuation and stop words."""

    text = text.lower()

    # Remove email addresses and web links.
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " ",
        text
    )
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove phone-like number sequences.
    text = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
        " ",
        text
    )

    # Keep letters and digits; replace punctuation with spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# --------------------------------------------------
# 4. SKILL MATCHING AND EXPERIENCE EXTRACTION
# --------------------------------------------------

def normalize_for_matching(text):
    """Normalize punctuation and spacing for phrase matching."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text, phrase):
    """Check a skill phrase without matching partial words."""

    normalized_text = normalize_for_matching(text)
    normalized_phrase = normalize_for_matching(phrase)

    if not normalized_phrase:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(normalized_phrase)
        + r"(?!\w)"
    )

    return re.search(pattern, normalized_text) is not None


def find_skills(text):
    """Return required skills detected in the original text."""

    matched = []

    for skill in REQUIRED_SKILLS:
        possible_names = [skill] + SKILL_ALIASES.get(skill, [])

        if any(contains_phrase(text, name) for name in possible_names):
            matched.append(skill)

    return matched


def extract_years_experience(text):
    """Extract an approximate years-of-experience value."""

    patterns = [
        # Example: 2.5 years of experience
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
        r"(?:\s+of)?\s+(?:relevant\s+)?experience",

        # Example: Experience: 2 years
        r"experience\s*[:\-]\s*(\d+(?:\.\d+)?)"
        r"\s*\+?\s*(?:years?|yrs?)",

        # Example: 2 years of experience
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
        r"\s+experience",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None


# --------------------------------------------------
# 5. CREATE TEXT EMBEDDINGS
# --------------------------------------------------

def create_embeddings(documents):
    """
    Try Sentence Transformers first.
    Fall back to TF-IDF if the model cannot be loaded.
    """

    try:
        from sentence_transformers import SentenceTransformer

        print("\nLoading Sentence Transformer model...")
        print("The first run may download the model.")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print("Using Sentence Transformers.")
        return embeddings, "Sentence Transformers"

    except Exception as error:
        print("\nSentence Transformers unavailable.")
        print(f"Reason: {error}")
        print("Falling back to TF-IDF...")

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )

        embeddings = vectorizer.fit_transform(documents)

        print("Using TF-IDF fallback.")
        return embeddings, "TF-IDF"


# --------------------------------------------------
# 6. SCORE AND RANK CANDIDATES
# --------------------------------------------------

def min_max_scale(values):
    """Scale values to the 0-1 range."""

    values = pd.Series(values, dtype=float)
    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return pd.Series([0.0] * len(values), index=values.index)

    return (values - minimum) / (maximum - minimum)


def main():
    print("=" * 60)
    print("AI-BASED RESUME SCREENING SYSTEM")
    print("=" * 60)

    if not JOB_DESCRIPTION_PATH.exists():
        raise FileNotFoundError(
            f"Job description not found: {JOB_DESCRIPTION_PATH}"
        )

    if not RESUME_DIR.exists():
        raise FileNotFoundError(
            f"Resume folder not found: {RESUME_DIR}"
        )

    job_description = extract_text(JOB_DESCRIPTION_PATH)

    if not job_description.strip():
        raise ValueError("The job description file is empty.")

    # Find TXT, PDF and DOCX resumes.
    supported_extensions = {".txt", ".pdf", ".docx"}

    resume_files = sorted(
        file for file in RESUME_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in supported_extensions
    )

    if not resume_files:
        raise ValueError(
            "No TXT, PDF or DOCX resumes found in data/resumes."
        )

    candidates = []

    for file_path in resume_files:
        original_text = extract_text(file_path)

        if not original_text.strip():
            print(f"Skipping empty/unreadable resume: {file_path.name}")
            continue

        cleaned_text = preprocess_text(original_text)
        matched_skills = find_skills(original_text)
        missing_skills = [
            skill for skill in REQUIRED_SKILLS
            if skill not in matched_skills
        ]

        candidates.append({
            "resume_file": file_path.name,
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "years_experience": extract_years_experience(
                original_text
            ),
            "skill_coverage": (
                len(matched_skills) / len(REQUIRED_SKILLS)
            ),
        })

    if not candidates:
        raise ValueError(
            "No readable resumes were found. Check the resume files."
        )

    cleaned_job_description = preprocess_text(job_description)

    documents = [cleaned_job_description] + [
        candidate["cleaned_text"]
        for candidate in candidates
    ]

    embeddings, embedding_method = create_embeddings(documents)

    # Compare every resume with the job description.
    similarities = cosine_similarity(
        embeddings[0:1],
        embeddings[1:]
    )[0]

    for candidate, similarity in zip(candidates, similarities):
        candidate["semantic_similarity"] = float(similarity)

    results = pd.DataFrame(candidates)

    # Min-max scaling makes the semantic score relative to this batch.
    results["semantic_scaled"] = min_max_scale(
        results["semantic_similarity"]
    )

    results["final_score"] = (
        SEMANTIC_WEIGHT * results["semantic_scaled"]
        + SKILL_WEIGHT * results["skill_coverage"]
    ) * 100

    results = results.sort_values(
        by=["final_score", "semantic_similarity"],
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1)
    )

    results["matched_skills"] = results["matched_skills"].apply(
        lambda skills: ", ".join(skills)
    )
    results["missing_skills"] = results["missing_skills"].apply(
        lambda skills: ", ".join(skills)
    )

    # Save a report without the full resume text.
    csv_columns = [
        "rank",
        "resume_file",
        "semantic_similarity",
        "skill_coverage",
        "final_score",
        "matched_skills",
        "missing_skills",
        "years_experience",
    ]

    csv_path = RESULTS_DIR / "ranking.csv"
    results[csv_columns].to_csv(csv_path, index=False)

    # Create a ranking chart.
    chart_data = results.sort_values("final_score", ascending=True)

    plt.figure(figsize=(11, max(5, len(chart_data) * 0.55)))
    plt.barh(
        chart_data["resume_file"],
        chart_data["final_score"]
    )
    plt.xlabel("Final Score (0-100)")
    plt.ylabel("Resume")
    plt.title("Resume Ranking for Machine Learning Engineer (NLP)")
    plt.xlim(0, 100)
    plt.tight_layout()

    chart_path = RESULTS_DIR / "ranking.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print(f"Embedding method: {embedding_method}")
    print(f"Resumes processed: {len(results)}")
    print("=" * 60)

    display_columns = [
        "rank",
        "resume_file",
        "final_score",
        "skill_coverage",
        "years_experience",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            formatters={
                "final_score": "{:.2f}".format,
                "skill_coverage": "{:.1%}".format,
            },
        )
    )

    print("\nResults saved:")
    print(f"CSV report: {csv_path}")
    print(f"Ranking chart: {chart_path}")

    print(
        "\nNote: Scores are relative to this resume batch. "
        "They are not probabilities of being hired."
    )


if __name__ == "__main__":
    main()
