"""
Pharmacy AI Assistant - Vector Search Ingestion Pipeline
Embeds documents using OpenAI and ingests them into Databricks Vector Search.
Uses REST API directly for Python 3.14 compatibility.
"""

import os
import time
import logging
import sys
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/vector_ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────
DATABRICKS_HOST    = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN   = os.getenv("DATABRICKS_TOKEN")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
VS_ENDPOINT        = os.getenv("DATABRICKS_VS_ENDPOINT", "pharmacy-ai-endpoint")
CATALOG            = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA             = os.getenv("DATABRICKS_SCHEMA", "pharmacy_ai")

EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
BATCH_SIZE         = 100       # documents per embedding batch
MAX_DOCS_PER_TYPE  = 11000     # full ingestion

PREPARED_DIR       = Path("../data/prepared")
TABLE_NAME         = f"{CATALOG}.{SCHEMA}.pharmacy_documents"
INDEX_NAME         = f"{CATALOG}.{SCHEMA}.pharmacy_index"


# ── Clients ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
ws_client     = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
vs_client     = VectorSearchClient(
    workspace_url=DATABRICKS_HOST,
    personal_access_token=DATABRICKS_TOKEN,
    disable_notice=True,
)


# ── Embedding ──────────────────────────────────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def embed_dataframe(df: pd.DataFrame, text_col: str = "rag_text") -> pd.DataFrame:
    """Add embedding column to dataframe in batches."""
    texts     = df[text_col].tolist()
    all_embeddings = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"  Embedding {len(texts):,} documents in {total_batches} batches...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        try:
            embeddings = embed_batch(batch)
            all_embeddings.extend(embeddings)

            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"  Batch {batch_num}/{total_batches} complete")

            time.sleep(0.1)   # rate limit safety

        except Exception as e:
            logger.error(f"  Embedding batch {batch_num} failed: {e}")
            raise

    df = df.copy()
    df["embedding"] = all_embeddings
    return df


# ── Databricks Setup ───────────────────────────────────────────────────────
def ensure_schema_exists():
    """Create catalog schema if it doesn't exist."""
    try:
        ws_client.schemas.get(full_name=f"{CATALOG}.{SCHEMA}")
        logger.info(f"  Schema {CATALOG}.{SCHEMA} already exists")
    except Exception:
        logger.info(f"  Creating schema {CATALOG}.{SCHEMA}...")
        ws_client.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        logger.info(f"  Schema created")


DB_HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json",
}


def sql_execute(statement: str, warehouse_id: str, max_retries: int = 3) -> dict:
    """Execute SQL via Databricks Statement Execution API with retry logic."""
    url     = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload = {"statement": statement, "warehouse_id": warehouse_id, "wait_timeout": "50s"}

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=DB_HEADERS, json=payload, timeout=60)
            if not r.ok:
                logger.error(f"  SQL Error {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
            result = r.json()
            status = result.get("status", {}).get("state", "")
            if status == "FAILED":
                msg = result.get("status", {}).get("error", {}).get("message", "Unknown")
                raise RuntimeError(f"SQL failed: {msg}")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"  Attempt {attempt + 1} failed, retrying in {wait}s: {type(e).__name__}")
                time.sleep(wait)
            else:
                logger.error(f"  All {max_retries} attempts failed")
                raise


def get_warehouse_id() -> str:
    """Get the first available SQL warehouse ID."""
    url = f"{DATABRICKS_HOST}/api/2.0/sql/warehouses"
    r   = requests.get(url, headers=DB_HEADERS, timeout=30)
    r.raise_for_status()
    warehouses = r.json().get("warehouses", [])
    if not warehouses:
        raise RuntimeError("No SQL warehouses found.")
    wh_id = warehouses[0]["id"]
    logger.info(f"  Using warehouse: {warehouses[0]['name']} ({wh_id})")
    return wh_id


CHECKPOINT_FILE = Path("../logs/ingestion_checkpoint.json")


def load_checkpoint() -> int:
    """Load last successfully inserted document index."""
    if CHECKPOINT_FILE.exists():
        import json
        data = json.loads(CHECKPOINT_FILE.read_text())
        idx  = data.get("last_inserted", 0)
        logger.info(f"  Checkpoint found — resuming from document {idx:,}")
        return idx
    return 0


def save_checkpoint(inserted: int):
    """Save progress checkpoint to disk."""
    import json
    CHECKPOINT_FILE.write_text(json.dumps({"last_inserted": inserted}))


def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def upload_to_delta(df: pd.DataFrame, warehouse_id: str):
    """Upload embedded documents to Databricks Delta table with checkpoint support."""
    logger.info(f"  Uploading {len(df):,} documents to Delta table...")

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id       STRING NOT NULL,
            doc_type STRING,
            source   STRING,
            rag_text STRING,
            embedding ARRAY<FLOAT>
        )
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """
    sql_execute(create_sql, warehouse_id)
    logger.info(f"  Delta table ready: {TABLE_NAME}")

    batch_size = 25
    total      = len(df)
    start_from = load_checkpoint()
    inserted   = start_from

    for i in range(start_from, total, batch_size):
        batch  = df.iloc[i : i + batch_size]
        values = []

        for idx, row in batch.iterrows():
            text_escaped = str(row["rag_text"])[:2000].replace("'", "''").replace("\\", "\\\\")
            emb_str      = ", ".join(str(round(x, 6)) for x in row["embedding"])
            values.append(
                f"('{idx}', '{row.get('doc_type','unknown')}', "
                f"'{row.get('source','unknown')}', "
                f"'{text_escaped}', array({emb_str}))"
            )

        insert_sql = f"INSERT INTO {TABLE_NAME} VALUES {', '.join(values)}"
        sql_execute(insert_sql, warehouse_id)
        inserted += len(batch)
        save_checkpoint(inserted)

        if inserted % 100 == 0 or inserted >= total:
            logger.info(f"  Inserted {inserted:,}/{total:,} documents")

        time.sleep(0.5)  # rate limit safety

    clear_checkpoint()
    logger.info(f"  Upload complete: {inserted:,} records")


def create_vector_index():
    """Create Vector Search index on the Delta table."""
    logger.info(f"  Creating Vector Search index: {INDEX_NAME}...")

    try:
        vs_client.create_delta_sync_index(
            endpoint_name        = VS_ENDPOINT,
            index_name           = INDEX_NAME,
            source_table_name    = TABLE_NAME,
            pipeline_type        = "TRIGGERED",
            primary_key          = "id",
            embedding_dimension  = EMBEDDING_DIM,
            embedding_vector_column = "embedding",
        )
        logger.info(f"  Index created: {INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"  Index already exists — skipping")
        else:
            raise


# ── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("  PHARMACY AI - Vector Ingestion Pipeline")
    logger.info("=" * 60)

    # Verify prepared data exists
    for f in ["prepared_medicines.csv", "prepared_interactions.csv", "prepared_reviews.csv"]:
        if not (PREPARED_DIR / f).exists():
            logger.error(f"Missing: {PREPARED_DIR / f} — run data_preparation.py first")
            sys.exit(1)

    # Load prepared datasets (limited for first run)
    logger.info(f"\nLoading prepared datasets (max {MAX_DOCS_PER_TYPE:,} per type)...")

    medicines    = pd.read_csv(PREPARED_DIR / "prepared_medicines.csv").head(MAX_DOCS_PER_TYPE)
    interactions = pd.read_csv(PREPARED_DIR / "prepared_interactions.csv").head(MAX_DOCS_PER_TYPE)
    reviews      = pd.read_csv(PREPARED_DIR / "prepared_reviews.csv").head(MAX_DOCS_PER_TYPE)

    logger.info(f"  Medicines:    {len(medicines):,}")
    logger.info(f"  Interactions: {len(interactions):,}")
    logger.info(f"  Reviews:      {len(reviews):,}")

    # Embed all documents
    logger.info("\nGenerating embeddings...")
    logger.info("Embedding medicines...")
    medicines    = embed_dataframe(medicines)
    logger.info("Embedding interactions...")
    interactions = embed_dataframe(interactions)
    logger.info("Embedding reviews...")
    reviews      = embed_dataframe(reviews)

    # Combine all documents
    all_docs = pd.concat([medicines, interactions, reviews], ignore_index=True)
    logger.info(f"\nTotal documents embedded: {len(all_docs):,}")

    # Setup Databricks schema
    logger.info("\nSetting up Databricks...")
    ensure_schema_exists()

    # Get warehouse
    logger.info("\nConnecting to SQL warehouse...")
    warehouse_id = get_warehouse_id()

    # Upload to Delta table
    logger.info("\nUploading to Delta table...")
    upload_to_delta(all_docs, warehouse_id)

    # Create Vector Search index
    logger.info("\nCreating Vector Search index...")
    create_vector_index()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("  INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Documents ingested: {len(all_docs):,}")
    logger.info(f"  Delta table:        {TABLE_NAME}")
    logger.info(f"  Vector index:       {INDEX_NAME}")
    logger.info(f"  Elapsed time:       {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()