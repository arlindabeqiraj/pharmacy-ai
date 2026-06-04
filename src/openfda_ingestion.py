"""
Pharmacy AI Assistant - OpenFDA Automatic Ingestion
Fetches official FDA drug labels for missing drugs and
adds them directly to Databricks Vector Search.
"""

import os
import time
import logging
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient

load_dotenv(dotenv_path=Path("../.env"))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/openfda_ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

CATALOG          = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA           = os.getenv("DATABRICKS_SCHEMA", "pharmacy_ai")
VS_ENDPOINT      = os.getenv("DATABRICKS_VS_ENDPOINT", "pharmacy-ai-endpoint")

TABLE_NAME       = f"{CATALOG}.{SCHEMA}.pharmacy_documents"
INDEX_NAME       = f"{CATALOG}.{SCHEMA}.pharmacy_index"
EMBEDDING_MODEL  = "text-embedding-3-small"
EMBEDDING_DIM    = 1536

OPENFDA_URL      = "https://api.fda.gov/drug/label.json"
BATCH_SIZE       = 25

DB_HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json",
}

# ── Priority drugs missing from current dataset ────────────────────────────
PRIORITY_DRUGS = [
    "warfarin",          "azithromycin",      "lisinopril",
    "metoprolol",        "levothyroxine",     "simvastatin",
    "losartan",          "gabapentin",        "sertraline",
    "fluoxetine",        "clopidogrel",       "prednisone",
    "tramadol",          "clonazepam",        "hydrochlorothiazide",
    "furosemide",        "albuterol",         "digoxin",
    "ciprofloxacin",     "doxycycline",       "hydrocodone",
    "acetaminophen",     "naproxen",          "diazepam",
    "propranolol",       "ramipril",          "insulin glargine",
    "atorvastatin",      "rosuvastatin",      "esomeprazole",
]

# ── Clients ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vs_client     = VectorSearchClient(
    workspace_url=DATABRICKS_HOST,
    personal_access_token=DATABRICKS_TOKEN,
    disable_notice=True,
)


# ── OpenFDA Fetcher ────────────────────────────────────────────────────────
def fetch_fda_label(drug_name: str) -> dict | None:
    """Fetch official FDA drug label from OpenFDA API."""
    try:
        r = requests.get(
            OPENFDA_URL,
            params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 1},
            timeout=10,
        )
        if r.status_code != 200:
            # Try brand name search
            r = requests.get(
                OPENFDA_URL,
                params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 1},
                timeout=10,
            )

        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0]

        logger.warning(f"  No FDA data found for: {drug_name}")
        return None

    except Exception as e:
        logger.error(f"  FDA fetch failed for {drug_name}: {e}")
        return None


def extract_fda_info(label: dict, drug_name: str) -> str:
    """Extract relevant fields from FDA label and build RAG text."""
    def get_field(label: dict, *keys) -> str:
        for key in keys:
            val = label.get(key, [])
            if val and isinstance(val, list) and val[0]:
                return str(val[0])[:600].strip()
        return "Not available"

    openfda      = label.get("openfda", {})
    brand_names  = openfda.get("brand_name", [drug_name.title()])
    generic_name = openfda.get("generic_name", [drug_name.title()])
    drug_class   = openfda.get("pharm_class_epc", ["Not specified"])

    purpose       = get_field(label, "purpose", "indications_and_usage")
    warnings      = get_field(label, "warnings", "warnings_and_cautions")
    side_effects  = get_field(label, "adverse_reactions", "side_effects")
    dosage        = get_field(label, "dosage_and_administration")
    interactions  = get_field(label, "drug_interactions")
    contraindic   = get_field(label, "contraindications")

    return (
        f"FDA DRUG LABEL — {brand_names[0]}\n"
        f"Generic Name: {generic_name[0]}\n"
        f"Drug Class: {drug_class[0]}\n"
        f"Purpose/Indications: {purpose}\n"
        f"Warnings: {warnings}\n"
        f"Side Effects: {side_effects}\n"
        f"Dosage: {dosage}\n"
        f"Drug Interactions: {interactions}\n"
        f"Contraindications: {contraindic}\n"
        f"Source: FDA Official Label (OpenFDA)"
    )


# ── Embedding ──────────────────────────────────────────────────────────────
def embed_texts(texts: list) -> list:
    """Embed texts using OpenAI."""
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


# ── Databricks SQL ─────────────────────────────────────────────────────────
def sql_execute(statement: str, warehouse_id: str, max_retries: int = 3) -> dict:
    """Execute SQL via Databricks REST API with retry."""
    url     = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload = {"statement": statement, "warehouse_id": warehouse_id, "wait_timeout": "50s"}

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=DB_HEADERS, json=payload, timeout=60)
            if not r.ok:
                logger.error(f"  SQL Error {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
            result = r.json()
            if result.get("status", {}).get("state") == "FAILED":
                raise RuntimeError(result.get("status", {}).get("error", {}).get("message", ""))
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"  Retry {attempt+1} in {wait}s: {type(e).__name__}")
                time.sleep(wait)
            else:
                raise


def get_warehouse_id() -> str:
    """Get first available SQL warehouse."""
    r = requests.get(f"{DATABRICKS_HOST}/api/2.0/sql/warehouses", headers=DB_HEADERS, timeout=30)
    r.raise_for_status()
    warehouses = r.json().get("warehouses", [])
    if not warehouses:
        raise RuntimeError("No SQL warehouses found.")
    return warehouses[0]["id"]


def upload_documents(docs: list, warehouse_id: str):
    """Upload FDA documents to Delta table in batches."""
    total    = len(docs)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch  = docs[i : i + BATCH_SIZE]
        values = []

        for doc in batch:
            text_escaped = doc["rag_text"].replace("'", "''").replace("\\", "\\\\")
            emb_str      = ", ".join(str(round(x, 6)) for x in doc["embedding"])
            values.append(
                f"('{doc['id']}', '{doc['doc_type']}', '{doc['source']}', "
                f"'{text_escaped}', array({emb_str}))"
            )

        insert_sql = f"INSERT INTO {TABLE_NAME} VALUES {', '.join(values)}"
        sql_execute(insert_sql, warehouse_id)
        inserted += len(batch)
        logger.info(f"  Inserted {inserted}/{total} FDA documents")
        time.sleep(0.5)


# ── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("  PHARMACY AI - OpenFDA Ingestion Pipeline")
    logger.info(f"  Fetching {len(PRIORITY_DRUGS)} priority drugs from FDA")
    logger.info("=" * 60)

    # Step 1 — Fetch FDA labels
    logger.info("\nFetching FDA drug labels...")
    fda_docs = []

    for i, drug in enumerate(PRIORITY_DRUGS, 1):
        logger.info(f"  [{i:02d}/{len(PRIORITY_DRUGS)}] Fetching: {drug}...")
        label = fetch_fda_label(drug)
        if label:
            rag_text = extract_fda_info(label, drug)
            fda_docs.append({
                "drug_name": drug,
                "rag_text":  rag_text,
            })
            logger.info(f"  ✓ Got FDA label for {drug}")
        time.sleep(0.3)  # rate limit

    logger.info(f"\n  Fetched {len(fda_docs)}/{len(PRIORITY_DRUGS)} FDA labels")

    if not fda_docs:
        logger.error("No FDA labels fetched. Check internet connection.")
        sys.exit(1)

    # Step 2 — Generate embeddings
    logger.info("\nGenerating embeddings...")
    texts      = [doc["rag_text"] for doc in fda_docs]
    embeddings = embed_texts(texts)

    for i, doc in enumerate(fda_docs):
        doc["embedding"] = embeddings[i]
        doc["id"]        = f"fda_{doc['drug_name'].replace(' ', '_')}_{i+10000}"
        doc["doc_type"]  = "fda_label"
        doc["source"]    = "openfda_official"

    logger.info(f"  Embedded {len(fda_docs)} documents")

    # Step 3 — Upload to Databricks
    logger.info("\nConnecting to Databricks...")
    warehouse_id = get_warehouse_id()
    logger.info(f"  Warehouse ID: {warehouse_id}")

    logger.info("\nUploading to Delta table...")
    upload_documents(fda_docs, warehouse_id)

    # Step 4 — Sync Vector Search index
    logger.info("\nSyncing Vector Search index...")
    try:
        idx = vs_client.get_index(VS_ENDPOINT, INDEX_NAME)
        idx.sync()
        logger.info("  ✓ Vector Search sync triggered")
    except Exception as e:
        logger.warning(f"  Sync warning: {e}")

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("  OPENFDA INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  FDA labels ingested: {len(fda_docs)}")
    logger.info(f"  Drugs covered:       {', '.join(d['drug_name'] for d in fda_docs[:5])}...")
    logger.info(f"  Elapsed time:        {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()