"""
Pharmacy AI Assistant - Model Deployment Script
Demonstrates the canonical Databricks deployment path:

  Step 1: mlflow.pyfunc.log_model()     — log to MLflow
  Step 2: mlflow.register_model()       — register to Unity Catalog
  Step 3: Model Serving endpoint        — deploy via Databricks UI or SDK

D10: Deployment, Security & Governance
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("../.env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def deploy():
    """
    Full deployment pipeline:
    1. Configure MLflow to use Databricks as tracking server
    2. Log PharmacyRAGModel as MLflow PyFunc
    3. Register to Unity Catalog
    4. Print instructions for Model Serving endpoint creation
    """
    import mlflow
    from mlflow_model import log_pharmacy_model

    # ── Step 1: Configure MLflow tracking ─────────────────────────────────
    databricks_host  = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    catalog          = os.getenv("DATABRICKS_CATALOG", "workspace")
    schema           = os.getenv("DATABRICKS_SCHEMA",  "pharmacy_ai")

    if databricks_host and databricks_token:
        # Use Databricks as MLflow tracking server
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        logger.info(f"MLflow tracking: Databricks ({databricks_host})")
    else:
        # Fall back to local MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        logger.info("MLflow tracking: local (mlflow.db)")

    mlflow.set_experiment("/pharmacy-ai-deployment")

    # ── Step 2 & 3: Log + Register ─────────────────────────────────────────
    model_name = f"{catalog}.{schema}.pharmacy_rag_model"

    logger.info("=" * 55)
    logger.info("  PHARMACY AI — Model Deployment")
    logger.info("=" * 55)
    logger.info(f"  Model name : {model_name}")

    run_id = log_pharmacy_model(
        run_name   = "pharmacy-rag-v1",
        model_name = model_name,
        register   = True,
    )

    # ── Step 4: Serving endpoint instructions ──────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  NEXT STEP — Create Model Serving Endpoint")
    logger.info("=" * 55)
    logger.info("  Option A — Databricks UI:")
    logger.info(f"    1. Go to {databricks_host}/ml/endpoints")
    logger.info("    2. Click 'Create serving endpoint'")
    logger.info(f"    3. Select model: {model_name}")
    logger.info("    4. Choose compute size and scale-to-zero")
    logger.info("")
    logger.info("  Option B — Databricks SDK (programmatic):")
    logger.info("    from databricks.sdk import WorkspaceClient")
    logger.info("    w = WorkspaceClient()")
    logger.info("    w.serving_endpoints.create(")
    logger.info(f'      name="pharmacy-ai-endpoint",')
    logger.info("      config=ServedModelInput(")
    logger.info(f'        model_name="{model_name}",')
    logger.info('        model_version="1",')
    logger.info('        workload_size="Small",')
    logger.info('        scale_to_zero_enabled=True,')
    logger.info("      )")
    logger.info("    )")
    logger.info("")
    logger.info(f"  Run ID: {run_id}")
    logger.info("=" * 55)

    return run_id


if __name__ == "__main__":
    deploy()