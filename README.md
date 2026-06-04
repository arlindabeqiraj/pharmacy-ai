# Pharmacy AI Assistant

An AI-powered pharmaceutical assistant that answers questions about medications, side effects, and drug interactions. Built with a dual-mode architecture: a RAG pipeline for fast, precise answers and a LangChain ReAct agent for complex multi-part queries.

## Features

- **RAG Pipeline** — Multi-query retrieval, drug-aware reranking, and input/output guardrails
- **LangChain ReAct Agent** — Autonomous multi-step reasoning with three specialized tools
- **Safety Guardrails** — Two-layer system (regex + LLM classifier) that blocks harmful queries and off-topic questions
- **Confidence Scoring** — High / Medium / Low confidence with source citation on every agent response
- **MLflow Integration** — Experiment tracking, model registry, and Unity Catalog deployment
- **RAG / Agent Toggle** — Switch between modes directly in the UI

## Architecture

```
User query
    ↓
┌─────────────────────────────────────────────┐
│  RAG Mode (/ask)          Agent Mode (/agent/ask)  │
│                                             │
│  LLM Analyzer             ReAct Loop        │
│  Input Guardrail    →     rag_search        │
│  Multi-Query Retrieval    drug_info         │
│  Reranking                interaction_check │
│  Chunk Filter                               │
│  LLM Generation           Final Answer      │
│  Output Guardrail                           │
└─────────────────────────────────────────────┘
    ↓
Databricks Vector Search (~81K chunks)
```

## Tech Stack

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| LLM                 | Groq Llama 3.3 70B                       |
| Embeddings          | OpenAI text-embedding-3-small (1536 dim) |
| Vector Store        | Databricks Vector Search — Delta Sync    |
| Agent Framework     | LangChain ReAct (langchain_classic)      |
| Tracking & Registry | MLflow + Unity Catalog                   |
| Backend             | FastAPI                                  |
| Frontend            | React 18 + Vite                          |

## Datasets

| Dataset              | Source           | Content                                          |
| -------------------- | ---------------- | ------------------------------------------------ |
| Medicine_Details.csv | Kaggle           | ~11,000 medications — uses, side effects, dosage |
| ddinter              | Kaggle / DDInter | Drug-drug interactions with severity levels      |
| drugsComTest         | Kaggle           | Patient reviews and experiences                  |
| OpenFDA API          | FDA.gov          | Official FDA drug labels                         |

## Project Structure

```
pharmacy-ai/
├── src/
│   ├── rag_pipeline.py       # RAG pipeline — LLM-driven, 8-step
│   ├── pharmacy_agent.py     # LangChain ReAct Agent + 3 tools
│   ├── main.py               # FastAPI — /ask, /agent/ask, /mlflow
│   ├── mlflow_model.py       # MLflow PyFunc model wrapper
│   ├── deploy_model.py       # Unity Catalog deployment script
│   ├── data_preparation.py   # Data cleaning + chunking
│   ├── vector_ingestion.py   # Embed + upload to Databricks VS
│   ├── openfda_ingestion.py  # OpenFDA API ingestion
│   └── rag_evaluation.py     # MLflow evaluation metrics
├── frontend/                 # React UI with RAG/Agent toggle
├── data/
│   ├── raw/                  # Original CSV datasets
│   └── prepared/             # Processed chunks
├── eval/                     # Evaluation results
├── logs/                     # Application logs
└── config/.env               # API keys
```

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

Configure `config/.env`:

```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_token
DATABRICKS_VS_ENDPOINT=pharmacy-ai-endpoint
DATABRICKS_CATALOG=workspace
DATABRICKS_SCHEMA=pharmacy_ai
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

## Run

```bash
# Backend
cd src && python main.py

# Frontend (separate terminal)
cd frontend && npm run dev

# Deploy model to Unity Catalog
cd src && python deploy_model.py
```

## API Endpoints

| Endpoint                 | Description                             |
| ------------------------ | --------------------------------------- |
| `POST /ask`              | RAG pipeline — fast, direct retrieval   |
| `POST /agent/ask`        | ReAct agent — multi-step reasoning      |
| `POST /mlflow/log-model` | Log and register model to Unity Catalog |
| `GET /health`            | Health check                            |
