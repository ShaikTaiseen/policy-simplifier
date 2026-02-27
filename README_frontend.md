# Health Insurance Policy Simplifier Frontend

This Streamlit app provides a frontend for the #3 project: upload policy PDF, ask coverage questions, and view citation-backed answers.

## Quick Start

1. Install dependencies:
   - `pip install -r requirements_frontend.txt`
2. Run app:
   - `streamlit run frontend_app.py`

## Modes

- **Live API mode** (default): connects to backend URL (default `http://localhost:8000`)
- **Demo mode**: works without backend so you can still show UI flow in hackathon demo

## Expected Backend APIs

The frontend will try these endpoints automatically:

- Upload endpoints (multipart): `/upload`, `/policies/upload`, `/ingest`
- Query endpoints (JSON): `/query`, `/ask`, `/qa`
- Health check: `/health`

### Upload request

- form-data `file`: PDF file
- form-data `policy_name`: string

### Upload response (any one key works)

```json
{
  "policy_id": "policy_123"
}
```

### Query request

```json
{
  "question": "Is this surgery covered?",
  "policy_id": "policy_123"
}
```

### Query response (flexible keys supported)

```json
{
  "verdict": "Yes",
  "answer": "Covered after waiting period.",
  "confidence": 0.82,
  "reasoning": "Clause indicates inclusion post waiting period.",
  "citations": [
    {
      "source": "Policy A",
      "page": 12,
      "section": "Maternity Benefits",
      "text": "..."
    }
  ]
}
```

## Safety Note

Use as decision-support only. Final claim approval remains with insurer/TPA.
