import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional

import fitz
import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    FAISS = None
    Document = None
    HuggingFaceEmbeddings = None
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


app = FastAPI(title="Health Insurance Policy Simplifier API", version="1.0.0")
POLICY_NOTICE = "This assistant supports policy interpretation only and does not provide legal advice."
QUESTION_TYPES = {"FACTUAL", "ANALYTICAL", "COVERAGE / DECISION"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


@dataclass
class Chunk:
    page_no: int
    section: str
    text: str
    embedding: Optional[List[float]] = None


class QueryRequest(BaseModel):
    question: str
    policy_id: Optional[str] = None
    explain_language: str = "English"


class CompareRequest(BaseModel):
    question: str
    policy_ids: List[str]
    explain_language: str = "English"


class FeedbackRequest(BaseModel):
    query_id: str
    is_correct: bool


class ClaimPredictionRequest(BaseModel):
    policy_id: str
    procedure: str
    claim_amount: float
    waiting_period_completed_months: int = 0
    has_pre_existing_condition: bool = False


POLICIES: Dict[str, Dict] = {}
QUERY_LOG: List[Dict] = []
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

KG_TERM_GROUPS = {
    "benefit": [
        "maternity",
        "hospitalization",
        "cashless",
        "room rent",
        "icu",
        "day care",
        "ambulance",
    ],
    "exclusion": [
        "excluded",
        "exclusion",
        "not covered",
        "non payable",
        "cosmetic",
        "dental",
        "infertility",
    ],
    "condition": [
        "pre-existing",
        "waiting period",
        "co-payment",
        "deductible",
        "sub-limit",
        "pre-authorization",
    ],
    "procedure": [
        "surgery",
        "knee replacement",
        "cataract",
        "dialysis",
        "chemotherapy",
        "cardiac",
    ],
}


def _get_openai_client():
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(page_text: str, page_no: int, chunk_size: int = 700, overlap: int = 120) -> List[Chunk]:
    text = _normalize_whitespace(page_text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        section = "Policy Clause"
        chunks.append(Chunk(page_no=page_no, section=section, text=chunk_text))
        if end == len(text):
            break
        start = max(end - overlap, 0)

    return chunks


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


def _embed_texts(client, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _score_chunk(question_tokens: List[str], chunk_text: str) -> float:
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    chunk_counter = Counter(chunk_tokens)
    return float(sum(chunk_counter.get(token, 0) for token in question_tokens))


def _retrieve_top_chunks(question: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
    question_tokens = _tokenize(question)
    if not question_tokens:
        return []

    ranked = sorted(
        ((chunk, _score_chunk(question_tokens, chunk.text)) for chunk in chunks),
        key=lambda item: item[1],
        reverse=True,
    )
    filtered = [chunk for chunk, score in ranked if score > 0]
    return filtered[:top_k]


def _extract_pdf_text_pages(file_bytes: bytes) -> Dict:
    pages: List[str] = []
    parser_used = "pymupdf"

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_index in range(len(doc)):
            pages.append(doc.load_page(page_index).get_text("text") or "")
    except Exception:
        pages = []

    total_chars = sum(len((page or "").strip()) for page in pages)
    if total_chars >= 200:
        return {"pages": pages, "parser_used": parser_used}

    parser_used = "pdfplumber"
    fallback_pages: List[str] = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                fallback_pages.append(page.extract_text() or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF with fallbacks: {exc}")

    if sum(len((page or "").strip()) for page in fallback_pages) == 0:
        raise HTTPException(status_code=400, detail="No readable text found in PDF")

    return {"pages": fallback_pages, "parser_used": parser_used}


def _build_policy_knowledge_graph(policy_id: str, policy_name: str, chunks: List[Chunk]) -> Dict:
    if nx is None:
        return {"nodes": [], "edges": [], "available": False}

    graph = nx.Graph()
    graph.add_node(policy_id, label=policy_name, kind="policy")

    for chunk in chunks:
        chunk_lower = chunk.text.lower()
        found_terms: List[Dict] = []
        for group, terms in KG_TERM_GROUPS.items():
            for term in terms:
                if term in chunk_lower:
                    node_id = f"{group}:{term}"
                    found_terms.append({"node_id": node_id, "label": term, "kind": group})

        seen_ids = set()
        dedup_terms = []
        for item in found_terms:
            if item["node_id"] not in seen_ids:
                dedup_terms.append(item)
                seen_ids.add(item["node_id"])

        for item in dedup_terms:
            graph.add_node(item["node_id"], label=item["label"], kind=item["kind"])
            if graph.has_edge(policy_id, item["node_id"]):
                graph[policy_id][item["node_id"]]["weight"] += 1
            else:
                graph.add_edge(policy_id, item["node_id"], relation="mentions", weight=1)

        for i in range(len(dedup_terms)):
            for j in range(i + 1, len(dedup_terms)):
                left = dedup_terms[i]["node_id"]
                right = dedup_terms[j]["node_id"]
                if graph.has_edge(left, right):
                    graph[left][right]["weight"] += 1
                else:
                    graph.add_edge(left, right, relation="co_occurs", weight=1)

    nodes = []
    for node_id, data in graph.nodes(data=True):
        if node_id == policy_id:
            continue
        degree = int(graph.degree(node_id))
        nodes.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "kind": data.get("kind", "unknown"),
            "degree": degree,
        })

    edges = []
    for source, target, data in graph.edges(data=True):
        if source == policy_id or target == policy_id:
            continue
        edges.append({
            "source": source,
            "target": target,
            "relation": data.get("relation", "co_occurs"),
            "weight": int(data.get("weight", 1)),
        })

    nodes = sorted(nodes, key=lambda item: item["degree"], reverse=True)[:40]
    edges = sorted(edges, key=lambda item: item["weight"], reverse=True)[:80]

    return {"nodes": nodes, "edges": edges, "available": True}


def _predict_claim_approval(
    verdict: str,
    confidence: float,
    claim_amount: float,
    waiting_period_completed_months: int,
    has_pre_existing_condition: bool,
) -> Dict:
    score = 0.5
    factors: List[str] = []

    verdict_upper = verdict.upper()
    if verdict_upper == "SUPPORTED":
        score += 0.25
        factors.append("Coverage verdict is supported by policy clauses")
    elif verdict_upper == "PARTIALLY ADDRESSED":
        score += 0.05
        factors.append("Coverage is partial; additional checks may be required")
    elif verdict_upper == "NOT MENTIONED":
        score -= 0.2
        factors.append("Procedure is not explicitly mentioned")
    else:
        score -= 0.1
        factors.append("Coverage verdict is unclear")

    score += (confidence - 0.5) * 0.4

    if waiting_period_completed_months >= 24:
        score += 0.07
        factors.append("Waiting period appears sufficiently completed")
    elif waiting_period_completed_months < 12:
        score -= 0.08
        factors.append("Waiting period may be insufficient")

    if has_pre_existing_condition:
        score -= 0.08
        factors.append("Pre-existing condition may trigger exclusions")

    if claim_amount > 500000:
        score -= 0.06
        factors.append("High claim amount may require stricter review")
    elif claim_amount <= 100000:
        score += 0.03
        factors.append("Moderate claim amount improves approval likelihood")

    probability = float(max(0.05, min(0.95, score)))

    risk_band = "Low"
    if probability < 0.4:
        risk_band = "High"
    elif probability < 0.7:
        risk_band = "Medium"

    return {
        "approval_probability": probability,
        "risk_band": risk_band,
        "factors": factors,
    }


def _get_langchain_embeddings():
    if EMBEDDING_PROVIDER == "local":
        if HuggingFaceEmbeddings is None:
            return None, "local_unavailable"
        model = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        return model, "local_bge"

    if OpenAIEmbeddings is None:
        return None, "openai_unavailable"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "openai_key_missing"

    model = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return model, "openai_embedding"


def _build_documents(policy_name: str, chunks: List[Chunk]) -> List:
    if Document is None:
        return []

    docs = []
    for index, chunk in enumerate(chunks, start=1):
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": policy_name,
                    "page": chunk.page_no,
                    "section": chunk.section,
                    "chunk_id": index,
                },
            )
        )
    return docs


def _retrieve_top_documents(question: str, policy: Dict, top_k: int = 3) -> List:
    vectorstore = policy.get("vectorstore")
    if vectorstore is not None:
        try:
            return vectorstore.similarity_search(question, k=top_k)
        except Exception:
            pass

    fallback_chunks = _retrieve_top_chunks(question, policy.get("chunks", []), top_k=top_k)
    docs = []
    for index, chunk in enumerate(fallback_chunks, start=1):
        docs.append(
            {
                "page_content": chunk.text,
                "metadata": {
                    "source": policy.get("policy_name", "Policy Document"),
                    "page": chunk.page_no,
                    "section": chunk.section,
                    "chunk_id": index,
                },
            }
        )
    return docs


def _get_llm_chain_client():
    if ChatOpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=api_key)


def _normalize_retrieved_docs(retrieved_docs: List) -> List[Dict]:
    normalized: List[Dict] = []
    for doc in retrieved_docs:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            normalized.append({"page_content": doc.page_content, "metadata": dict(doc.metadata or {})})
        elif isinstance(doc, dict):
            normalized.append(
                {
                    "page_content": doc.get("page_content", ""),
                    "metadata": dict(doc.get("metadata", {})),
                }
            )
    return normalized


def _safe_json_parse(text: str) -> Optional[Dict]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _heuristic_verdict(question: str, context: str) -> str:
    q = question.lower()
    c = context.lower()

    exclusion_signals = ["not covered", "excluded", "exclusion", "not payable", "inadmissible"]
    inclusion_signals = ["covered", "admissible", "payable", "eligible", "shall be covered"]
    conditional_signals = ["subject to", "waiting period", "pre-authorization", "as per terms"]

    has_exclusion = any(term in c for term in exclusion_signals)
    has_inclusion = any(term in c for term in inclusion_signals)
    has_conditional = any(term in c for term in conditional_signals)

    if has_exclusion and not has_inclusion:
        return "No"
    if has_inclusion and not has_exclusion and not has_conditional:
        return "Yes"
    if has_inclusion or has_exclusion or has_conditional:
        return "Unclear"

    if any(term in q for term in ["covered", "coverage", "eligible", "claim"]):
        return "Unclear"
    return "Unclear"


def _classify_question_type(question: str) -> str:
    q = question.lower()

    coverage_terms = [
        "covered",
        "coverage",
        "allowed",
        "eligible",
        "guaranteed",
        "claim",
        "payable",
    ]
    analytical_terms = ["how", "why", "impact", "reason", "rationale", "addresses"]

    if any(term in q for term in coverage_terms):
        return "COVERAGE / DECISION"
    if any(term in q for term in analytical_terms):
        return "ANALYTICAL"
    return "FACTUAL"


def _allowed_verdicts(question_type: str) -> List[str]:
    if question_type == "FACTUAL":
        return ["SUPPORTED", "PARTIALLY ADDRESSED", "NOT MENTIONED"]
    if question_type == "ANALYTICAL":
        return ["SUPPORTED", "PARTIALLY ADDRESSED", "UNCLEAR"]
    return ["SUPPORTED", "UNCLEAR", "NOT MENTIONED"]


def _format_answer_block(verdict: str, confidence_percent: int, explanation_bullets: List[str]) -> str:
    safe_bullets = explanation_bullets or ["No sufficient document evidence found."]
    bullets_text = "\n".join(f"- {bullet}" for bullet in safe_bullets)
    return (
        "Answer:\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence_percent}%\n"
        "Explanation:\n"
        f"{bullets_text}"
    )


def _calibrate_confidence_percent(
    base_confidence_percent: int,
    verdict: str,
    mode: str,
    question_type: str,
    citations_count: int,
) -> int:
    confidence = int(max(0, min(100, base_confidence_percent)))

    if citations_count == 0:
        return min(confidence, 55)

    if mode == "rag_gpt4o_mini":
        confidence += 5

    if citations_count >= 2:
        confidence += 5
    if citations_count >= 3:
        confidence += 3

    verdict_upper = verdict.upper()
    if verdict_upper == "SUPPORTED":
        confidence += 4
    elif verdict_upper in {"NOT MENTIONED", "UNCLEAR"}:
        confidence -= 4

    if question_type == "FACTUAL" and verdict_upper == "SUPPORTED":
        confidence += 3

    if mode == "keyword_fallback":
        confidence = min(confidence, 72)

    return int(max(35, min(95, confidence)))


def _generate_rag_answer(
    client,
    question: str,
    retrieved_docs: List[Dict],
    question_type: str,
    explain_language: str,
) -> Dict:
    context_parts = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.get("metadata", {})
        page_no = metadata.get("page", "-")
        section = metadata.get("section", "Policy Clause")
        content = doc.get("page_content", "")
        context_parts.append(
            f"[C{idx}] Page {page_no} | {section}: {content[:1200]}"
        )
    context = "\n\n".join(context_parts)

    allowed_verdicts = " | ".join(_allowed_verdicts(question_type))

    system_prompt = (
        "You are an insurance policy interpretation assistant. Use only the provided policy context. "
        "Do not provide legal advice. "
        "Never hallucinate or assume information. "
        "Never fabricate clauses or citations. "
        "Keep reasoning explainable and directly tied to context. Return strict JSON only."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Question Type: {question_type}\n"
        f"Allowed Verdicts: {allowed_verdicts}\n\n"
        f"Policy Context:\n{context}\n\n"
        "Rules:\n"
        "1) Answer strictly from document context.\n"
        "2) For FACTUAL questions, do not default to uncertainty when explicit factual text exists.\n"
        "3) Be conservative for COVERAGE / DECISION questions.\n"
        "4) If evidence is missing, use the appropriate low-certainty verdict from allowed verdicts.\n\n"
        f"5) Add one plain-language explanation in {explain_language}. Keep it simple and short.\n\n"
        "Return JSON with keys exactly:\n"
        "question_type, verdict, confidence_percent, explanation_bullets, explanation_local\n"
        "where explanation_bullets is an array of concise strings grounded in the context."
    )

    response = client.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    raw_text = response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json_parse(raw_text)
    if parsed is None:
        parsed = {
            "question_type": question_type,
            "verdict": "UNCLEAR" if question_type == "ANALYTICAL" else "NOT MENTIONED",
            "confidence_percent": 45,
            "explanation_bullets": ["Model response was not in strict JSON format."],
            "explanation_local": "Plain-language explanation unavailable.",
        }

    parsed_question_type = str(parsed.get("question_type", question_type)).upper()
    if parsed_question_type not in QUESTION_TYPES:
        parsed_question_type = question_type

    verdict = str(parsed.get("verdict", "UNCLEAR")).upper()
    allowed = _allowed_verdicts(parsed_question_type)
    if verdict not in allowed:
        verdict = "UNCLEAR" if "UNCLEAR" in allowed else "NOT MENTIONED"

    confidence_raw = parsed.get("confidence_percent", 50)
    try:
        confidence_percent = int(confidence_raw)
    except (TypeError, ValueError):
        confidence_percent = 50
    confidence_percent = int(max(0, min(100, confidence_percent)))

    explanation_bullets = parsed.get("explanation_bullets", [])
    if not isinstance(explanation_bullets, list):
        explanation_bullets = [str(explanation_bullets)]
    explanation_bullets = [str(item).strip() for item in explanation_bullets if str(item).strip()]
    if not explanation_bullets:
        explanation_bullets = ["No sufficient document evidence found."]

    answer_block = _format_answer_block(verdict, confidence_percent, explanation_bullets)

    return {
        "question_type": parsed_question_type,
        "verdict": verdict,
        "answer": answer_block,
        "confidence": confidence_percent / 100.0,
        "confidence_percent": confidence_percent,
        "reasoning": "\n".join(f"- {bullet}" for bullet in explanation_bullets),
        "explanation_local": parsed.get("explanation_local") or "Plain-language explanation unavailable.",
    }


def _record_query_event(result: Dict, question: str, policy_id: str) -> None:
    QUERY_LOG.append(
        {
            "query_id": result["query_id"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "policy_id": policy_id,
            "policy_name": POLICIES.get(policy_id, {}).get("policy_name", policy_id),
            "verdict": result.get("verdict"),
            "mode": result.get("mode"),
            "grounded": bool(result.get("grounded", False)),
            "citations_count": len(result.get("citations", [])),
            "has_valid_citations": any((c.get("text") or "").strip() for c in result.get("citations", [])),
            "is_correct": None,
        }
    )


def _answer_with_policy(question: str, policy_id: str, explain_language: str) -> Dict:
    policy = POLICIES.get(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail=f"policy_id not found: {policy_id}")

    question_type = _classify_question_type(question)
    llm_client = _get_llm_chain_client()
    best_docs = _normalize_retrieved_docs(_retrieve_top_documents(question, policy, top_k=3))

    if not best_docs:
        if question_type == "FACTUAL":
            verdict = "NOT MENTIONED"
            confidence_percent = 90
        elif question_type == "ANALYTICAL":
            verdict = "UNCLEAR"
            confidence_percent = 55
        else:
            verdict = "NOT MENTIONED"
            confidence_percent = 85

        explanation_bullets = ["No relevant document statements were retrieved for this question."]
        result = {
            "query_id": f"qry_{uuid.uuid4().hex[:10]}",
            "question_type": question_type,
            "verdict": verdict,
            "answer": _format_answer_block(verdict, confidence_percent, explanation_bullets),
            "confidence": confidence_percent / 100.0,
            "confidence_percent": confidence_percent,
            "reasoning": "- Could not retrieve relevant clauses from available policy documents.",
            "mode": "keyword_fallback",
            "embedding_mode": policy.get("embedding_mode", "unknown"),
            "vector_store": "faiss" if policy.get("vectorstore") is not None else "none",
            "parser_used": policy.get("parser_used", "unknown"),
            "grounded": False,
            "policy_notice": POLICY_NOTICE,
            "explanation_local": "Plain-language explanation unavailable due to missing evidence.",
            "citations": [],
        }
        _record_query_event(result, question, policy_id)
        return result

    if llm_client is not None:
        try:
            generated = _generate_rag_answer(llm_client, question, best_docs, question_type, explain_language)
            question_type = generated["question_type"]
            verdict = generated["verdict"]
            answer = generated["answer"]
            confidence = generated["confidence"]
            confidence_percent = generated["confidence_percent"]
            reasoning = generated["reasoning"]
            explanation_local = generated.get("explanation_local", "Plain-language explanation unavailable.")
            mode = "rag_gpt4o_mini"
        except Exception:
            context_text = " ".join(doc.get("page_content", "") for doc in best_docs)
            legacy_verdict = _heuristic_verdict(question, context_text)
            if legacy_verdict == "Yes":
                verdict = "SUPPORTED"
                confidence = 0.78
                confidence_percent = 78
            elif legacy_verdict == "No":
                verdict = "NOT MENTIONED" if question_type != "ANALYTICAL" else "UNCLEAR"
                confidence = 0.7
                confidence_percent = 70
            else:
                verdict = "PARTIALLY ADDRESSED" if question_type != "COVERAGE / DECISION" else "UNCLEAR"
                confidence = 0.6
                confidence_percent = 60
            answer = _format_answer_block(
                verdict,
                confidence_percent,
                ["Fallback mode due to LLM call failure.", "Result derived from retrieved clauses only."],
            )
            reasoning = "- Fallback mode due to LLM call failure.\n- Result derived from retrieved clauses only."
            explanation_local = "Plain-language explanation unavailable in fallback mode."
            mode = "keyword_fallback"
    else:
        context_text = " ".join(doc.get("page_content", "") for doc in best_docs)
        legacy_verdict = _heuristic_verdict(question, context_text)
        if legacy_verdict == "Yes":
            verdict = "SUPPORTED"
            confidence = 0.78
            confidence_percent = 78
        elif legacy_verdict == "No":
            verdict = "NOT MENTIONED" if question_type != "ANALYTICAL" else "UNCLEAR"
            confidence = 0.7
            confidence_percent = 70
        else:
            verdict = "PARTIALLY ADDRESSED" if question_type != "COVERAGE / DECISION" else "UNCLEAR"
            confidence = 0.6
            confidence_percent = 60
        answer = _format_answer_block(
            verdict,
            confidence_percent,
            ["Keyword retrieval mode used.", "Set OPENAI_API_KEY for full RAG with gpt-4o-mini."],
        )
        reasoning = "Keyword retrieval mode (set OPENAI_API_KEY for full RAG with gpt-4o-mini)."
        explanation_local = "Plain-language explanation unavailable in fallback mode."
        mode = "keyword_fallback"

    citations = [
        {
            "source": doc.get("metadata", {}).get("source", policy.get("policy_name", policy_id)),
            "page": doc.get("metadata", {}).get("page", "-"),
            "section": doc.get("metadata", {}).get("section", "Policy Clause"),
            "text": doc.get("page_content", "")[:350],
        }
        for doc in best_docs
    ]

    confidence_percent = _calibrate_confidence_percent(
        base_confidence_percent=confidence_percent,
        verdict=verdict,
        mode=mode,
        question_type=question_type,
        citations_count=len(citations),
    )
    confidence = confidence_percent / 100.0

    if answer.startswith("Answer:\n"):
        answer_lines = answer.splitlines()
        for idx, line in enumerate(answer_lines):
            if line.startswith("Confidence:"):
                answer_lines[idx] = f"Confidence: {confidence_percent}%"
                break
        answer = "\n".join(answer_lines)

    if len(citations) == 0:
        verdict = "UNCLEAR" if question_type == "ANALYTICAL" else "NOT MENTIONED"
        confidence = min(confidence, 0.55)
        confidence_percent = min(confidence_percent, 55)
        answer = _format_answer_block(
            verdict,
            confidence_percent,
            ["No citations available; cannot provide a grounded conclusive answer."],
        )
        reasoning = "- No citations available; answer forced to low-certainty verdict."

    result = {
        "query_id": f"qry_{uuid.uuid4().hex[:10]}",
        "question_type": question_type,
        "verdict": verdict,
        "answer": answer,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "reasoning": reasoning,
        "mode": mode,
        "embedding_mode": policy.get("embedding_mode", "unknown"),
        "vector_store": "faiss" if policy.get("vectorstore") is not None else "none",
        "parser_used": policy.get("parser_used", "unknown"),
        "grounded": len(citations) > 0,
        "policy_notice": POLICY_NOTICE,
        "explanation_local": explanation_local,
        "citations": citations,
    }
    _record_query_event(result, question, policy_id)
    return result


@app.get("/health")
def health_check():
    return {"status": "ok", "policies_loaded": len(POLICIES)}


@app.post("/upload")
async def upload_policy(file: UploadFile = File(...), policy_name: str = Form("")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_MB} MB.",
        )

    extraction = _extract_pdf_text_pages(file_bytes)
    page_texts = extraction["pages"]
    parser_used = extraction["parser_used"]

    chunks: List[Chunk] = []
    for page_index, page_text in enumerate(page_texts):
        chunks.extend(_split_into_chunks(page_text, page_no=page_index + 1))

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in PDF")

    embeddings_model, embedding_mode = _get_langchain_embeddings()
    policy_title = policy_name.strip() or file.filename
    documents = _build_documents(policy_title, chunks)

    vectorstore = None
    if embeddings_model is not None and FAISS is not None and documents:
        try:
            vectorstore = FAISS.from_documents(documents, embeddings_model)
        except Exception:
            vectorstore = None

    policy_id = f"policy_{uuid.uuid4().hex[:10]}"
    POLICIES[policy_id] = {
        "policy_id": policy_id,
        "policy_name": policy_title,
        "chunks": chunks,
        "documents": documents,
        "vectorstore": vectorstore,
        "filename": file.filename,
        "embedding_mode": embedding_mode,
        "parser_used": parser_used,
    }

    kg = _build_policy_knowledge_graph(policy_id, policy_title, chunks)
    POLICIES[policy_id]["knowledge_graph"] = kg

    return {
        "policy_id": policy_id,
        "policy_name": POLICIES[policy_id]["policy_name"],
        "pages": len(page_texts),
        "chunks": len(chunks),
        "parser_used": parser_used,
        "embedding_mode": embedding_mode,
        "vector_store": "faiss" if vectorstore is not None else "none",
        "rag_mode": "langchain_faiss" if vectorstore is not None else "keyword_fallback",
    }


@app.get("/knowledge_graph/{policy_id}")
def policy_knowledge_graph(policy_id: str):
    policy = POLICIES.get(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="policy_id not found")

    kg = policy.get("knowledge_graph") or {"nodes": [], "edges": [], "available": False}
    return {
        "policy_id": policy_id,
        "policy_name": policy.get("policy_name", policy_id),
        "available": kg.get("available", False),
        "nodes": kg.get("nodes", []),
        "edges": kg.get("edges", []),
    }


@app.post("/query")
def query_policy(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if not POLICIES:
        raise HTTPException(status_code=400, detail="No policies loaded. Upload a policy first.")

    if payload.policy_id:
        return _answer_with_policy(question, payload.policy_id, payload.explain_language)

    best_policy_id = None
    best_score = -1.0
    q_tokens = _tokenize(question)

    for candidate_policy_id, candidate_policy in POLICIES.items():
        top_docs = _normalize_retrieved_docs(_retrieve_top_documents(question, candidate_policy, top_k=3))
        score = 0.0
        for doc in top_docs:
            score += _score_chunk(q_tokens, doc.get("page_content", ""))
        if score > best_score:
            best_score = score
            best_policy_id = candidate_policy_id

    if best_policy_id is None:
        raise HTTPException(status_code=400, detail="No retrievable policies found")

    return _answer_with_policy(question, best_policy_id, payload.explain_language)


@app.post("/compare")
def compare_policies(payload: CompareRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    policy_ids = [item for item in payload.policy_ids if item in POLICIES]
    if len(policy_ids) != 2 or policy_ids[0] == policy_ids[1]:
        raise HTTPException(status_code=400, detail="Provide exactly two distinct valid policy_ids")

    first = _answer_with_policy(question, policy_ids[0], payload.explain_language)
    second = _answer_with_policy(question, policy_ids[1], payload.explain_language)

    return {
        "question": question,
        "comparison_mode": True,
        "policies": [
            {"policy_id": policy_ids[0], "policy_name": POLICIES[policy_ids[0]]["policy_name"]},
            {"policy_id": policy_ids[1], "policy_name": POLICIES[policy_ids[1]]["policy_name"]},
        ],
        "results": [first, second],
    }


@app.get("/evaluation")
def evaluation_summary():
    total_queries = len(QUERY_LOG)
    fallback_queries = sum(1 for item in QUERY_LOG if item.get("mode") == "keyword_fallback")
    grounded_queries = sum(1 for item in QUERY_LOG if item.get("grounded") is True)
    valid_citation_queries = sum(1 for item in QUERY_LOG if item.get("has_valid_citations") is True)

    reviewed = [item for item in QUERY_LOG if item.get("is_correct") is not None]
    correct = sum(1 for item in reviewed if item.get("is_correct") is True)

    return {
        "total_queries": total_queries,
        "reviewed_queries": len(reviewed),
        "accuracy": (correct / len(reviewed)) if reviewed else None,
        "citation_precision": (valid_citation_queries / grounded_queries) if grounded_queries else None,
        "fallback_rate": (fallback_queries / total_queries) if total_queries else 0.0,
        "recent_queries": QUERY_LOG[-20:],
    }


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    for item in reversed(QUERY_LOG):
        if item.get("query_id") == payload.query_id:
            item["is_correct"] = payload.is_correct
            return {"status": "updated", "query_id": payload.query_id, "is_correct": payload.is_correct}

    raise HTTPException(status_code=404, detail="query_id not found")


@app.post("/claim_prediction")
def claim_prediction(payload: ClaimPredictionRequest):
    procedure = payload.procedure.strip()
    if not procedure:
        raise HTTPException(status_code=400, detail="procedure is required")

    question = f"Is {procedure} covered under this policy?"
    answer_result = _answer_with_policy(question, payload.policy_id, "English")

    predicted = _predict_claim_approval(
        verdict=answer_result.get("verdict", "UNCLEAR"),
        confidence=float(answer_result.get("confidence", 0.5)),
        claim_amount=float(payload.claim_amount),
        waiting_period_completed_months=int(payload.waiting_period_completed_months),
        has_pre_existing_condition=bool(payload.has_pre_existing_condition),
    )

    return {
        "policy_id": payload.policy_id,
        "procedure": procedure,
        "coverage_verdict": answer_result.get("verdict"),
        "coverage_confidence": answer_result.get("confidence"),
        "approval_probability": predicted["approval_probability"],
        "risk_band": predicted["risk_band"],
        "factors": predicted["factors"],
        "grounded_citations": answer_result.get("citations", []),
    }
