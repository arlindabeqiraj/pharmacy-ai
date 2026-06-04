"""
Pharmacy AI Assistant - MLflow PyFunc Model
Wraps the RAG pipeline as an MLflow PyFunc model for deployment
via Databricks Model Serving (Unity Catalog).


"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── MLflow PyFunc Model ────────────────────────────────────────────────────
class PharmacyRAGModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for the Pharmacy AI RAG pipeline.

    Input schema:  DataFrame with column 'question' (str)
    Output schema: DataFrame with columns:
                   - answer (str)
                   - has_interaction (bool)
                   - chunks_used (int)
                   - elapsed_ms (int)
                   - sources (str, JSON-encoded list)

    Usage:
        model = mlflow.pyfunc.load_model("models:/pharmacy-ai/Production")
        result = model.predict(pd.DataFrame({"question": ["What is Ibuprofen?"]}))
    """

    def load_context(self, context):
        """
        Called once when the model is loaded.
        Imports the RAG pipeline — deferred to avoid import errors at log time.
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from rag_pipeline import ask
        self._ask = ask
        logger.info("PharmacyRAGModel: RAG pipeline loaded")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Run the RAG pipeline for each question in the input DataFrame.

        Args:
            model_input: DataFrame with column 'question'

        Returns:
            DataFrame with answer and metadata columns
        """
        import json

        if "question" not in model_input.columns:
            raise ValueError(
                "Input DataFrame must have a 'question' column. "
                f"Got columns: {list(model_input.columns)}"
            )

        results = []
        for question in model_input["question"]:
            response = self._ask(str(question))
            results.append({
                "answer":          response.answer,
                "has_interaction": response.has_interaction,
                "chunks_used":     response.chunks_used,
                "elapsed_ms":      response.elapsed_ms,
                "sources":         json.dumps(response.sources),
            })

        return pd.DataFrame(results)


# ── MLflow Signature ───────────────────────────────────────────────────────
def get_model_signature():
    """
    Define the MLflow model signature.
    Signature = input schema + output schema — required for Model Serving.
    """
    from mlflow.models import infer_signature

    sample_input = pd.DataFrame({
        "question": ["What are the side effects of Ibuprofen?"]
    })
    sample_output = pd.DataFrame({
        "answer":          ["Ibuprofen side effects include..."],
        "has_interaction": [False],
        "chunks_used":     [5],
        "elapsed_ms":      [3000],
        "sources":         ['["Ibuprofen 400mg"]'],
    })

    return infer_signature(sample_input, sample_output)


# ── Log Model to MLflow ────────────────────────────────────────────────────
def log_pharmacy_model(
    run_name:    str = "pharmacy-rag-model",
    model_name:  str = "pharmacy-ai",
    register:    bool = False,
):
    """
    Log the PharmacyRAGModel to MLflow and optionally register to Unity Catalog.

    Args:
        run_name:   MLflow run name
        model_name: Model name in Unity Catalog (catalog.schema.model_name)
        register:   If True, register to Unity Catalog after logging

    Returns:
        run_id: MLflow run ID
    """
    signature = get_model_signature()

    with mlflow.start_run(run_name=run_name) as run:
        # Log model parameters
        mlflow.log_params({
            "model_type":      "RAG",
            "llm":             "llama-3.3-70b-versatile",
            "embedding_model": "text-embedding-3-small",
            "vector_store":    "databricks-vector-search",
            "pipeline_steps":  "guardrail→embed→retrieve→rerank→filter→generate→guardrail",
        })

        # Log the PyFunc model
        mlflow.pyfunc.log_model(
            artifact_path  = "pharmacy_rag",
            python_model   = PharmacyRAGModel(),
            signature      = signature,
            pip_requirements = [
                "openai>=1.0.0",
                "groq>=0.4.0",
                "databricks-vectorsearch>=0.22",
                "python-dotenv>=1.0.0",
                "pandas>=2.0.0",
            ],
            input_example  = pd.DataFrame({
                "question": ["Can I take Warfarin with Aspirin?"]
            }),
        )

        run_id = run.info.run_id
        logger.info(f"Model logged | run_id: {run_id}")

        # Register to Unity Catalog if requested
        if register:
            model_uri = f"runs:/{run_id}/pharmacy_rag"
            registered = mlflow.register_model(
                model_uri  = model_uri,
                name       = model_name,
            )
            logger.info(
                f"Model registered to Unity Catalog: "
                f"{model_name} v{registered.version}"
            )

        return run_id