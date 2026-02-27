# Health Insurance Policy Simplifier

## Problem Statement Addressed
Health insurance policy documents are long, technical, and hard for most people to interpret quickly. During claim planning, users often struggle to understand:

- Whether a procedure is likely covered
- Which clauses support or restrict coverage
- How waiting periods and pre-existing conditions affect approval chances

This creates confusion, delays, and poor decision-making. The project addresses this by turning policy PDFs into a grounded, explainable assistant that returns citation-backed answers in plain language.

## Proposed Solution and Approach
This project provides a two-part system:

1. Backend API (FastAPI)
2. Frontend App (Streamlit)

### Core Workflow
1. User uploads a policy PDF.
2. Backend extracts text from pages and splits it into chunks.
3. Chunks are indexed for retrieval (FAISS when available).
4. User asks a question.
5. Relevant policy chunks are retrieved and used to generate a grounded answer.
6. Response includes verdict, confidence, reasoning, and citations.
7. Optional claim prediction estimates approval probability using policy verdict + claim factors.

### Design Principles
- Grounded responses first (with citations)
- Plain-language output for accessibility
- Fallback behavior when LLM or external services are unavailable
- Fast hackathon-ready UX with practical features (compare mode, voice output, claim prediction)

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- LangChain (retrieval + orchestration)
- FAISS (vector store)
- PyMuPDF + pdfplumber (PDF extraction)
- OpenAI embeddings/chat (optional)
- HuggingFace local embeddings (optional)
- SQLite (state persistence)
- NetworkX (policy knowledge graph generation)

### Frontend
- Streamlit
- Requests
- gTTS (text-to-speech)

## Third-Party Resources Used
- OpenAI API (optional; for embeddings and LLM inference)
- HuggingFace model: BAAI/bge-small-en-v1.5 (optional local embedding)
- FAISS (vector similarity search)
- PyMuPDF and pdfplumber (document parsing)
- gTTS (voice playback)

## Setup and Run Instructions

### Prerequisites
- Python 3.10+ recommended
- Windows PowerShell (or any shell)

### 1) Install Backend Dependencies
From project root:

pip install -r requirements_backend.txt

### 2) Install Frontend Dependencies
From project root:

pip install -r requirements_frontend.txt

### 3) Optional Environment Variables
Set only what you need.

PowerShell examples:

$env:OPENAI_API_KEY="your_key_here"
$env:LLM_MODEL="gpt-4o-mini"
$env:EMBEDDING_PROVIDER="openai"   # or "local"
$env:EMBEDDING_MODEL="text-embedding-3-small"
$env:LOCAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
$env:APP_DB_PATH="insurance_app.db"
$env:MAX_UPLOAD_MB="25"

Notes:
- If OPENAI_API_KEY is not set, the app still runs in fallback mode.
- SQLite persistence is enabled by default and stores data in insurance_app.db.

### 4) Start Backend API
From project root:

uvicorn backend_app:app --host 127.0.0.1 --port 8000

Health check:

http://127.0.0.1:8000/health

### 5) Start Frontend
Open a second terminal in project root:

python -m streamlit run frontend_app.py --server.port 8501

Open:

http://127.0.0.1:8501

## Key API Endpoints
- GET /health
- POST /upload
- POST /query
- POST /compare
- POST /claim_prediction
- GET /knowledge_graph/{policy_id}
- GET /evaluation

## Demo Flow (Quick)
1. Launch backend and frontend.
2. Upload a policy PDF.
3. Ask a coverage question.
4. Review verdict, reasoning, and citations.
5. Use claim prediction with different amounts/conditions to compare risk.

## Current Scope and Limitations
- This is a decision-support tool, not legal or final claim approval advice.
- Output quality depends on policy text quality and retrieval relevance.
- For best answer quality, enable OpenAI-based mode with API key.

## Project Files
- backend_app.py: FastAPI backend logic
- frontend_app.py: Streamlit UI
- requirements_backend.txt: backend dependencies
- requirements_frontend.txt: frontend dependencies
- insurance_app.db: SQLite persistence file
