"""
Pharmacy AI Assistant - Data Preparation Pipeline
Prepares Medicine Details, Drug Interactions, and Patient Reviews for RAG ingestion.
"""

import pandas as pd
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# ── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/data_preparation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("../data/raw")
OUTPUT_DIR = Path("../data/prepared")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEDICINE_FILE     = DATA_DIR / "Medicine_Details.csv"
INTERACTIONS_FILE = DATA_DIR / "ddinter_downloads_code_A.csv"
REVIEWS_FILE      = DATA_DIR / "drugsComTest_raw.csv"

REVIEW_MIN_LENGTH = 100   # characters
REVIEW_MIN_USEFUL = 5     # usefulCount threshold
REVIEW_MAX_CHARS  = 500   # truncate long reviews


# ── Data Classes ───────────────────────────────────────────────────────────
@dataclass
class DatasetStats:
    name: str
    raw_count: int
    clean_count: int
    dropped: int
    columns: list

    def summary(self) -> str:
        pct = (self.dropped / self.raw_count * 100) if self.raw_count else 0
        return (
            f"  {self.name:<20} raw={self.raw_count:>7,} | "
            f"clean={self.clean_count:>7,} | "
            f"dropped={self.dropped:>6,} ({pct:.1f}%)"
        )


# ── Text Builders ──────────────────────────────────────────────────────────
def build_medicine_text(row: pd.Series) -> str:
    """Structured RAG document for a single medicine."""
    return (
        f"MEDICINE: {row['Medicine Name']}\n"
        f"COMPOSITION: {row['Composition']}\n"
        f"USES: {row['Uses']}\n"
        f"SIDE EFFECTS: {row['Side_effects']}\n"
        f"MANUFACTURER: {row['Manufacturer']}\n"
        f"USER RATINGS: Excellent {row['Excellent Review %']}% | "
        f"Average {row['Average Review %']}% | "
        f"Poor {row['Poor Review %']}%"
    )


def build_interaction_text(row: pd.Series) -> str:
    """Structured RAG document for a drug-drug interaction."""
    level = row["Level"].upper()
    warning = {
        "MAJOR":    "MAJOR — Avoid combination. Risk of serious adverse effects.",
        "MODERATE": "MODERATE — Use with caution. Monitor patient closely.",
        "MINOR":    "MINOR — Minimal clinical significance.",
        "UNKNOWN":  "UNKNOWN — Insufficient data. Consult a pharmacist.",
    }.get(level, "UNKNOWN interaction level.")

    return (
        f"DRUG INTERACTION ALERT\n"
        f"Drug A: {row['Drug_A']}\n"
        f"Drug B: {row['Drug_B']}\n"
        f"Interaction Level: {row['Level']}\n"
        f"Clinical Warning: {warning}\n"
        f"Recommendation: Consult a pharmacist or physician before combining "
        f"{row['Drug_A']} with {row['Drug_B']}."
    )


def build_review_text(row: pd.Series) -> str:
    """Structured RAG document for a patient review."""
    clean_review = (
        str(row["review"])
        .replace("&#039;", "'")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .strip()
    )
    truncated = clean_review[:REVIEW_MAX_CHARS]
    if len(clean_review) > REVIEW_MAX_CHARS:
        truncated += "..."

    return (
        f"PATIENT REVIEW - {row['drugName']}\n"
        f"Condition Treated: {row['condition']}\n"
        f"Rating: {row['rating']}/10\n"
        f"Helpful votes: {int(row['usefulCount'])}\n"
        f"Experience: {truncated}"
    )


# ── Dataset Loaders ────────────────────────────────────────────────────────
def load_medicines(path: Path) -> tuple:
    logger.info("Loading Medicine Details dataset...")
    df = pd.read_csv(path)
    raw = len(df)

    required_cols = ["Medicine Name", "Uses", "Side_effects", "Composition"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in medicines: {missing}")

    df = df.dropna(subset=["Medicine Name", "Uses"])
    df = df.fillna("Not available")
    df["rag_text"] = df.apply(build_medicine_text, axis=1)
    df["source"]   = "medicine_details"
    df["doc_type"] = "medicine_info"

    stats = DatasetStats(
        name="Medicines",
        raw_count=raw,
        clean_count=len(df),
        dropped=raw - len(df),
        columns=list(df.columns),
    )
    logger.info(f"  Loaded {len(df):,} medicines")
    return df, stats


def load_interactions(path: Path) -> tuple:
    logger.info("Loading Drug-Drug Interactions dataset...")
    df = pd.read_csv(path)
    raw = len(df)

    df = df.dropna(subset=["Drug_A", "Drug_B", "Level"])
    df["Level"] = df["Level"].str.strip().str.capitalize()

    level_counts = df["Level"].value_counts().to_dict()
    for level, count in level_counts.items():
        logger.info(f"  {level:<12}: {count:>6,}")

    df["rag_text"] = df.apply(build_interaction_text, axis=1)
    df["source"]   = "ddinter_database"
    df["doc_type"] = "drug_interaction"

    stats = DatasetStats(
        name="Interactions",
        raw_count=raw,
        clean_count=len(df),
        dropped=raw - len(df),
        columns=list(df.columns),
    )
    logger.info(f"  Loaded {len(df):,} interactions")
    return df, stats


def load_reviews(path: Path) -> tuple:
    logger.info("Loading Patient Reviews dataset...")
    df = pd.read_csv(path)
    raw = len(df)

    df = df.dropna(subset=["review", "drugName", "condition"])
    df = df[df["review"].str.len() > REVIEW_MIN_LENGTH]
    df = df[df["usefulCount"] > REVIEW_MIN_USEFUL]
    df = df.drop_duplicates(subset=["review"])

    df["rag_text"] = df.apply(build_review_text, axis=1)
    df["source"]   = "drugs_com_reviews"
    df["doc_type"] = "patient_review"

    stats = DatasetStats(
        name="Reviews",
        raw_count=raw,
        clean_count=len(df),
        dropped=raw - len(df),
        columns=list(df.columns),
    )
    logger.info(f"  Loaded {len(df):,} reviews")
    return df, stats


# ── Validation ─────────────────────────────────────────────────────────────
def validate_output(df: pd.DataFrame, name: str) -> bool:
    """Basic sanity checks on the prepared dataframe."""
    errors = []
    if df.empty:
        errors.append("DataFrame is empty")
    if "rag_text" not in df.columns:
        errors.append("Missing 'rag_text' column")
    if df["rag_text"].isnull().any():
        errors.append(f"{df['rag_text'].isnull().sum()} null rag_text values")

    if errors:
        for e in errors:
            logger.warning(f"  WARN {name}: {e}")
        return False

    logger.info(f"  {name} validation passed")
    return True


# ── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("  PHARMACY AI - Data Preparation Pipeline")
    logger.info("=" * 60)

    # Verify input files exist
    for f in [MEDICINE_FILE, INTERACTIONS_FILE, REVIEWS_FILE]:
        if not f.exists():
            logger.error(f"File not found: {f}")
            sys.exit(1)

    # Load and process all datasets
    medicines_df,    med_stats = load_medicines(MEDICINE_FILE)
    interactions_df, int_stats = load_interactions(INTERACTIONS_FILE)
    reviews_df,      rev_stats = load_reviews(REVIEWS_FILE)

    # Validate outputs
    logger.info("\nValidating outputs...")
    validate_output(medicines_df,    "Medicines")
    validate_output(interactions_df, "Interactions")
    validate_output(reviews_df,      "Reviews")

    # Save prepared datasets
    logger.info("\nSaving prepared datasets...")
    medicines_df.to_csv(
        OUTPUT_DIR / "prepared_medicines.csv", index=False, encoding="utf-8"
    )
    interactions_df.to_csv(
        OUTPUT_DIR / "prepared_interactions.csv", index=False, encoding="utf-8"
    )
    reviews_df.to_csv(
        OUTPUT_DIR / "prepared_reviews.csv", index=False, encoding="utf-8"
    )

    # Save sample RAG documents for inspection
    sample_path = OUTPUT_DIR / "sample_rag_documents.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SAMPLE RAG DOCUMENTS - Pharmacy AI\n")
        f.write("=" * 60 + "\n\n")

        f.write("-- MEDICINE SAMPLE --\n")
        f.write(medicines_df["rag_text"].iloc[0] + "\n\n")

        f.write("-- INTERACTION SAMPLE (MAJOR) --\n")
        major = interactions_df[interactions_df["Level"] == "Major"]
        if not major.empty:
            f.write(major["rag_text"].iloc[0] + "\n\n")

        f.write("-- REVIEW SAMPLE --\n")
        f.write(reviews_df["rag_text"].iloc[0] + "\n\n")

    # Final report
    elapsed = time.time() - start
    total   = med_stats.clean_count + int_stats.clean_count + rev_stats.clean_count

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE - Summary")
    logger.info("=" * 60)
    for s in [med_stats, int_stats, rev_stats]:
        logger.info(s.summary())
    logger.info(f"  {'TOTAL RAG DOCUMENTS':<20} {total:>7,}")
    logger.info(f"  {'Elapsed time':<20} {elapsed:.2f}s")
    logger.info(f"  Output saved to: {OUTPUT_DIR.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()