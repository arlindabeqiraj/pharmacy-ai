"""
Pharmacy AI Assistant - FastAPI Backend
REST API that exposes the RAG pipeline for the React frontend.
"""

import os
import time
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path("../.env"))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    logger.info("=" * 50)
    logger.info("  PHARMACY AI - FastAPI Backend Starting")
    logger.info("=" * 50)
    from rag_pipeline import ask
    app.state.ask = ask
    logger.info("  RAG pipeline loaded successfully")
    logger.info("  API ready at http://localhost:8000")
    logger.info("=" * 50)
    yield
    logger.info("  Shutting down Pharmacy AI API...")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Pharmacy AI Assistant API",
    description = "RAG-powered pharmacy assistant using Databricks Vector Search + Groq LLaMA",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request/Response Models ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length = 3,
        max_length = 500,
        description = "Pharmacy question from user",
        example     = "What are the side effects of ibuprofen?",
    )
    top_k: Optional[int] = Field(
        default     = 5,
        ge          = 1,
        le          = 10,
        description = "Number of context chunks to retrieve",
    )


class AnswerResponse(BaseModel):
    question:        str
    answer:          str
    chunks_used:     int
    sources:         list
    has_interaction: bool
    elapsed_ms:      int
    status:          str = "success"


class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: float


class ErrorResponse(BaseModel):
    status:  str = "error"
    message: str
    code:    int


# ── Middleware — Request logging ───────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = int((time.time() - start) * 1000)
    logger.info(f"  {request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
    return response


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Pharmacy AI Assistant API", "docs": "/docs", "health": "/health"}


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    tags           = ["System"],
)
async def health():
    """Check if the API is running."""
    return HealthResponse(
        status    = "healthy",
        version   = "1.0.0",
        timestamp = time.time(),
    )


@app.post(
    "/ask",
    response_model = AnswerResponse,
    summary        = "Ask a pharmacy question",
    tags           = ["Pharmacy"],
    responses      = {
        200: {"description": "Successful answer"},
        400: {"description": "Invalid question"},
        500: {"description": "Internal server error"},
    },
)
async def ask_question(request: QuestionRequest):
    """
    Ask a pharmacy question and get a RAG-powered answer.

    The pipeline:
    1. Embeds the question using OpenAI
    2. Retrieves relevant drug information from Databricks Vector Search
    3. Generates a safe answer using Groq LLaMA 3.3 70B
    4. Applies safety guardrails
    """
    logger.info(f"  Question: {request.question[:80]}")

    try:
        response = app.state.ask(request.question)

        return AnswerResponse(
            question        = response.question,
            answer          = response.answer,
            chunks_used     = response.chunks_used,
            sources         = response.sources,
            has_interaction = response.has_interaction,
            elapsed_ms      = response.elapsed_ms,
            status          = "success",
        )

    except Exception as e:
        logger.error(f"  Error processing question: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to process question: {str(e)}",
        )


@app.get(
    "/history",
    summary = "Get recent queries (placeholder)",
    tags    = ["Pharmacy"],
)
async def get_history():
    """Placeholder for query history — will be implemented with a database."""
    return {"history": [], "message": "History feature coming soon"}


# ── Exception Handlers ─────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code = exc.status_code,
        content     = ErrorResponse(
            message = exc.detail,
            code    = exc.status_code,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"  Unhandled exception: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = ErrorResponse(
            message = "An unexpected error occurred. Please try again.",
            code    = 500,
        ).model_dump(),
    )

# ── MLflow model logging endpoint  ────────────────────────────────────
class MLflowLogRequest(BaseModel):
    model_name: Optional[str] = Field(
        default     = None,
        description = "Unity Catalog model name (catalog.schema.model). Uses .env defaults if not provided.",
        example     = "workspace.pharmacy_ai.pharmacy_rag_model",
    )
    register: bool = Field(
        default     = False,
        description = "Register model to Unity Catalog after logging",
    )


class MLflowLogResponse(BaseModel):
    status:     str
    run_id:     str
    model_name: str
    registered: bool
    message:    str


@app.post(
    "/mlflow/log-model",
    response_model = MLflowLogResponse,
    summary        = "Log RAG pipeline as MLflow PyFunc model",
    tags           = ["Deployment"],
)
async def log_mlflow_model(request: MLflowLogRequest):
    try:
        import mlflow
        import os
        from mlflow_model import log_pharmacy_model

        catalog    = os.getenv("DATABRICKS_CATALOG", "workspace")
        schema     = os.getenv("DATABRICKS_SCHEMA",  "pharmacy_ai")
        model_name = request.model_name or f"{catalog}.{schema}.pharmacy_rag_model"

        if os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment("/pharmacy-ai-deployment")

        run_id = log_pharmacy_model(
            run_name   = "pharmacy-rag-api-log",
            model_name = model_name,
            register   = request.register,
        )

        return MLflowLogResponse(
            status     = "success",
            run_id     = run_id,
            model_name = model_name,
            registered = request.register,
            message    = (
                f"Model logged successfully. "
                f"{'Registered to Unity Catalog. ' if request.register else ''}"
                f"Next step: create a Model Serving endpoint in Databricks UI."
            ),
        )

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Agent endpoint ───────────────────────────────────────────────
class AgentRequest(BaseModel):
    question: str = Field(
        ...,
        min_length  = 3,
        max_length  = 500,
        description = "Pharmacy question for the ReAct agent",
        example     = "What is Metformin used for and does it interact with Warfarin?",
    )


class AgentAnswerResponse(BaseModel):
    question:     str
    answer:       str
    tools_used:   list
    elapsed_ms:   int
    reasoning:    str
    confidence:   str
    sources_used: int
    status:       str = "success"


@app.post(
    "/agent/ask",
    response_model = AgentAnswerResponse,
    summary        = "Ask a question using the ReAct Agent",
    tags           = ["Agent"],
    responses      = {
        200: {"description": "Agent answer with confidence score"},
        500: {"description": "Agent error"},
    },
)
async def agent_ask(request: AgentRequest):
    """
    Ask a pharmacy question using the LangChain ReAct Agent.

    Demonstrates  (Agents, Tools & Multi-step Workflows):
    - ReAct framework: Reason + Act loop
    - Tools use direct Vector Search 
    - Confidence scoring: High / Medium / Low
    - Source citation: sources_used field

    
    """
    try:
        from pharmacy_agent import ask_agent
        response = ask_agent(request.question)

        return AgentAnswerResponse(
            question     = response.question,
            answer       = response.answer,
            tools_used   = response.tools_used,
            elapsed_ms   = response.elapsed_ms,
            reasoning    = response.reasoning,
            confidence   = response.confidence,
            sources_used = response.sources_used,
            status       = "success",
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,
        log_level = "info",
    )