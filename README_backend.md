# Backend API (Health Insurance Policy Simplifier)

This backend is a FastAPI service that matches your Streamlit frontend contract.

It supports:
- **RAG via LangChain**
- **Vector DB via FAISS**
- **PDF parsing with PyMuPDF + `pdfplumber` fallback**
- **Embeddings**: OpenAI `text-embedding-3-small` or local `BAAI/bge-small-en-v1.5`
- **LLM answers**: `gpt-4o-mini`

## Run

1. Install dependencies
   - `pip install -r requirements_backend.txt`
2. Set environment variable (for full RAG)
  - PowerShell: `$env:OPENAI_API_KEY="your_key_here"`
  - Optional: `$env:LLM_MODEL="gpt-4o-mini"`
  - Optional: `$env:EMBEDDING_PROVIDER="openai"` (default) or `"local"`
  - Optional (OpenAI mode): `$env:EMBEDDING_MODEL="text-embedding-3-small"`
  - Optional (local mode): `$env:LOCAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"`
  - Optional (SQLite path): `$env:APP_DB_PATH="insurance_app.db"`
3. Start server
   - `uvicorn backend_app:app --reload --host 0.0.0.0 --port 8000`

## Endpoints

- `GET /health`
- `POST /upload` (multipart PDF upload)
- `POST /query` (Q&A)
- `POST /compare` (same question across 2 policies)
- `GET /knowledge_graph/{policy_id}` (policy entity relationship graph)
- `POST /claim_prediction` (simple approval probability estimator)
- `GET /evaluation` (grounded-rate/citation/fallback dashboard data)

## Contract

### Upload request

- form-data `file` (PDF)
- form-data `policy_name` (optional string)

### Upload response

```json
{
  "policy_id": "policy_abc123",
  "policy_name": "Star Health Gold",
  "pages": 21,
  "chunks": 85,
  "parser_used": "pymupdf",
  "embedding_mode": "openai_embedding",
  "vector_store": "faiss",
  "rag_mode": "langchain_faiss"
}
```

### Query request

```json
{
  "question": "Is knee replacement covered after waiting period?",
  "policy_id": "policy_abc123"
}
```

### Query response

```json
{
  "question_type": "COVERAGE / DECISION",
  "verdict": "Yes",
  "answer": "Answer:\nVerdict: SUPPORTED\nConfidence: 82%\nExplanation:\n- ...",
  "confidence": 0.78,
  "confidence_percent": 78,
  "reasoning": "Answer generated from highest-matching policy clauses.",
  "mode": "rag_gpt4o_mini",
  "embedding_mode": "openai_embedding",
  "vector_store": "faiss",
  "parser_used": "pymupdf",
  "citations": [
    {
      "source": "Star Health Gold",
      "page": 12,
    For best results, use `OPENAI_API_KEY` so the API runs full LangChain RAG with FAISS retrieval and `gpt-4o-mini` generation.
      "text": "..."
    }
  ]
}
```

## Note

For best results, use `OPENAI_API_KEY` so the API runs true RAG with `gpt-4o-mini`.
