# Pharmacy AI Assistant

RAG-powered pharmacy assistant using Databricks Vector Search + Groq LLaMA 3.3.

## Project Structure

```
pharmacy-ai/
├── data/
│   ├── raw/          # Original datasets (CSV)
│   └── prepared/     # Processed chunks ready for RAG
├── src/
│   ├── data_preparation.py   # Data cleaning + chunking
│   ├── vector_ingestion.py   # Embed + upload to Databricks
│   ├── openfda_ingestion.py  # OpenFDA automatic fetch
│   ├── rag_pipeline.py       # RAG + reranking + guardrails
│   ├── rag_evaluation.py     # LLM-as-judge evaluation
│   └── main.py               # FastAPI backend
├── frontend/                 # React UI
├── eval/                     # Evaluation results
├── logs/                     # Application logs
├── config/                   # Configuration
└── .env                      # API keys (never commit)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Backend
python src/main.py

# Frontend
cd frontend && npm run dev

# Evaluation
python src/rag_evaluation.py
```

## Tech Stack

- **Vector DB**: Databricks Vector Search
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: Groq LLaMA 3.3 70B
- **Backend**: FastAPI
- **Frontend**: React + Vite
- **Tracking**: MLflow
