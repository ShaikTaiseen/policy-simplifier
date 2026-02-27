import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


st.set_page_config(
    page_title="Health Insurance Policy Simplifier",
    page_icon="🩺",
    layout="wide",
)


def _tts_lang_code(language_label: str) -> str:
    mapping = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
    }
    return mapping.get(language_label, "en")


@st.cache_data(show_spinner=False)
def _synthesize_audio(text: str, language_label: str) -> bytes:
    if gTTS is None:
        raise RuntimeError("gTTS package is not installed")

    lang_code = _tts_lang_code(language_label)
    tts = gTTS(text=text[:900], lang=lang_code)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


def _extract_voice_text(result: Dict[str, Any], preferred_language: str) -> str:
    def _is_technical(text: str) -> bool:
        t = text.lower()
        technical_terms = [
            "rag",
            "openai_api_key",
            "keyword retrieval",
            "fallback",
            "mode",
            "citation",
            "vector",
        ]
        return any(term in t for term in technical_terms)

    def _default_spoken(verdict: str, language_label: str) -> str:
        verdict_upper = (verdict or "UNCLEAR").upper()
        templates = {
            "English": {
                "SUPPORTED": "The document supports this request.",
                "PARTIALLY ADDRESSED": "The document partially addresses this request.",
                "NOT MENTIONED": "This point is not mentioned in the document.",
                "UNCLEAR": "The document evidence is not clear for this request.",
            },
            "Hindi": {
                "SUPPORTED": "दस्तावेज़ के अनुसार यह अनुरोध समर्थित है।",
                "PARTIALLY ADDRESSED": "दस्तावेज़ में इस अनुरोध का आंशिक उल्लेख है।",
                "NOT MENTIONED": "दस्तावेज़ में इस बिंदु का उल्लेख नहीं है।",
                "UNCLEAR": "इस अनुरोध के लिए दस्तावेज़ का प्रमाण स्पष्ट नहीं है।",
            },
            "Telugu": {
                "SUPPORTED": "పత్రంలో ఈ అభ్యర్థనకు మద్దతు ఉంది.",
                "PARTIALLY ADDRESSED": "పత్రంలో ఈ అభ్యర్థన కొంతవరకు మాత్రమే ఉంది.",
                "NOT MENTIONED": "ఈ విషయం పత్రంలో ప్రస్తావించబడలేదు.",
                "UNCLEAR": "ఈ అభ్యర్థనకు సంబంధించిన ఆధారాలు పత్రంలో స్పష్టంగా లేవు.",
            },
        }
        lang_templates = templates.get(language_label, templates["English"])
        return lang_templates.get(verdict_upper, lang_templates["UNCLEAR"])

    local_text = (result.get("explanation_local") or "").strip()
    if (
        local_text
        and "unavailable" not in local_text.lower()
        and not _is_technical(local_text)
    ):
        return local_text

    if preferred_language in {"Hindi", "Telugu"}:
        return _default_spoken(result.get("verdict", "UNCLEAR"), preferred_language)

    answer_text = (result.get("answer") or "").strip()
    if answer_text:
        bullets = []
        for line in answer_text.splitlines():
            line = line.strip()
            if line.startswith("- ") and not _is_technical(line):
                bullets.append(line[2:].strip())
        if bullets:
            return bullets[0]

    return _default_spoken(result.get("verdict", "UNCLEAR"), preferred_language)


def _render_voice_player(result: Dict[str, Any], preferred_language: str) -> None:
    voice_text = _extract_voice_text(result, preferred_language)
    if not voice_text:
        st.caption("Voice: No text available")
        return

    if gTTS is None:
        st.caption("Voice: install gTTS to enable audio playback")
        return

    try:
        audio_bytes = _synthesize_audio(voice_text, preferred_language)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception:
        st.caption("Voice output unavailable for this response")


def _normalize_citations(raw_citations: Any) -> List[Dict[str, Any]]:
    if not raw_citations:
        return []

    citations: List[Dict[str, Any]] = []
    if isinstance(raw_citations, list):
        for item in raw_citations:
            if isinstance(item, dict):
                citations.append(
                    {
                        "source": item.get("source") or item.get("policy") or "Policy Document",
                        "page": item.get("page") or item.get("page_no") or "-",
                        "section": item.get("section") or "-",
                        "text": item.get("text") or item.get("snippet") or "",
                    }
                )
            else:
                citations.append(
                    {"source": "Policy Document", "page": "-", "section": "-", "text": str(item)}
                )
    return citations


def _parse_query_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = payload.get("answer") or payload.get("final_answer") or "No answer returned"
    verdict = payload.get("verdict") or payload.get("decision") or "Unclear"
    confidence = payload.get("confidence")
    if confidence is None and payload.get("confidence_percent") is not None:
        confidence = payload.get("confidence_percent")

    if confidence is not None:
        try:
            confidence = float(confidence)
            if confidence > 1:
                confidence = confidence / 100.0
        except (TypeError, ValueError):
            confidence = None
    reasoning = payload.get("reasoning") or payload.get("explanation") or ""

    citations_raw = payload.get("citations")
    if citations_raw is None:
        citations_raw = payload.get("sources")

    return {
        "query_id": payload.get("query_id"),
        "answer": answer,
        "verdict": str(verdict).strip().title(),
        "question_type": payload.get("question_type", "-"),
        "confidence": confidence,
        "reasoning": reasoning,
        "explanation_local": payload.get("explanation_local", ""),
        "mode": payload.get("mode", "unknown"),
        "grounded": bool(payload.get("grounded", False)),
        "policy_notice": payload.get("policy_notice")
        or "This assistant supports policy interpretation only and does not provide legal advice.",
        "citations": _normalize_citations(citations_raw),
    }


def _candidate_base_urls(base_url: str) -> List[str]:
    base = base_url.rstrip("/")
    candidates = [base]
    if "localhost" in base:
        candidates.append(base.replace("localhost", "127.0.0.1"))
    return candidates


def _api_health_check(base_url: str) -> bool:
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.get(f"{candidate}/health", timeout=8)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def upload_policy(base_url: str, uploaded_file, policy_name: str) -> Optional[Dict[str, Any]]:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/pdf")}
    data = {"policy_name": policy_name.strip() or uploaded_file.name}

    last_error = None
    for candidate in _candidate_base_urls(base_url):
        for endpoint in ("/upload", "/policies/upload", "/ingest"):
            try:
                resp = requests.post(f"{candidate}{endpoint}", files=files, data=data, timeout=60)
                if resp.ok:
                    payload = resp.json()
                    policy_id = payload.get("policy_id") or payload.get("id") or payload.get("document_id")
                    if policy_id:
                        return {
                            "policy_id": policy_id,
                            "policy_name": payload.get("policy_name") or data["policy_name"],
                            "metadata": payload,
                        }
                last_error = f"{resp.status_code} {resp.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)

    st.error(f"Upload failed. Could not find a compatible upload endpoint. Last error: {last_error}")
    return None
def ask_question(base_url: str, question: str, policy_id: Optional[str], explain_language: str) -> Dict[str, Any]:
    request_body = {"question": question}
    if policy_id:
        request_body["policy_id"] = policy_id
    request_body["explain_language"] = explain_language

    last_error = None
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.post(f"{candidate}/query", json=request_body, timeout=40)
            if resp.ok:
                return _parse_query_response(resp.json())
            last_error = f"{resp.status_code} {resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)

    return {
        "query_id": None,
        "answer": "I could not reach a compatible Q&A endpoint. Switch to demo mode or align backend routes.",
        "verdict": "Unclear",
        "confidence": None,
        "reasoning": f"Last error: {last_error}",
        "explanation_local": "",
        "mode": "error",
        "grounded": False,
        "policy_notice": "This assistant supports policy interpretation only and does not provide legal advice.",
        "citations": [],
    }


def compare_question(base_url: str, question: str, policy_ids: List[str], explain_language: str) -> Dict[str, Any]:
    request_body = {
        "question": question,
        "policy_ids": policy_ids,
        "explain_language": explain_language,
    }
    last_error = None
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.post(f"{candidate}/compare", json=request_body, timeout=60)
            if resp.ok:
                payload = resp.json()
                return {
                    "comparison_mode": True,
                    "question": payload.get("question", question),
                    "policies": payload.get("policies", []),
                    "results": [_parse_query_response(item) for item in payload.get("results", [])],
                }
            last_error = f"{resp.status_code}: {resp.text[:300]}"
        except requests.RequestException as exc:
            last_error = str(exc)

    return {
        "comparison_mode": True,
        "question": question,
        "policies": [],
        "results": [],
        "error": last_error or "Unknown error",
    }


def get_evaluation(base_url: str) -> Optional[Dict[str, Any]]:
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.get(f"{candidate}/evaluation", timeout=20)
            if resp.ok:
                return resp.json()
        except requests.RequestException:
            continue
    return None


def submit_feedback(base_url: str, query_id: str, is_correct: bool) -> bool:
    if not query_id:
        return False
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.post(
                f"{candidate}/feedback",
                json={"query_id": query_id, "is_correct": is_correct},
                timeout=15,
            )
            if resp.ok:
                return True
        except requests.RequestException:
            continue
    return False


def predict_claim(
    base_url: str,
    policy_id: str,
    procedure: str,
    claim_amount: float,
    waiting_period_completed_months: int,
    has_pre_existing_condition: bool,
) -> Optional[Dict[str, Any]]:
    if not policy_id:
        return None
    payload = {
        "policy_id": policy_id,
        "procedure": procedure,
        "claim_amount": claim_amount,
        "waiting_period_completed_months": waiting_period_completed_months,
        "has_pre_existing_condition": has_pre_existing_condition,
    }
    for candidate in _candidate_base_urls(base_url):
        try:
            resp = requests.post(f"{candidate}/claim_prediction", json=payload, timeout=30)
            if resp.ok:
                return resp.json()
        except requests.RequestException:
            continue
    return None


def demo_answer(question: str) -> Dict[str, Any]:
    q = question.lower()

    if "maternity" in q:
        return {
            "query_id": "demo_qry_1",
            "answer": "Yes, maternity hospitalization is covered after waiting period completion.",
            "verdict": "Yes",
            "confidence": 0.86,
            "reasoning": "The policy includes maternity benefits only after the defined waiting period.",
            "explanation_local": "24 mahine waiting period ke baad maternity hospitalization cover hota hai.",
            "mode": "demo",
            "grounded": True,
            "policy_notice": "This assistant supports policy interpretation only and does not provide legal advice.",
            "citations": [
                {
                    "source": "Sample Policy A",
                    "page": 14,
                    "section": "Maternity Benefits",
                    "text": "Maternity expenses are admissible after a waiting period of 24 months.",
                }
            ],
        }

    if "knee" in q or "replacement" in q:
        return {
            "query_id": "demo_qry_2",
            "answer": "Likely covered if medically necessary and pre-authorization conditions are met.",
            "verdict": "Unclear",
            "confidence": 0.68,
            "reasoning": "Coverage depends on exclusion checks and prior approval requirements.",
            "explanation_local": "Yeh tabhi cover hoga jab medical necessity aur pre-authorization conditions meet ho.",
            "mode": "demo",
            "grounded": True,
            "policy_notice": "This assistant supports policy interpretation only and does not provide legal advice.",
            "citations": [
                {
                    "source": "Sample Policy A",
                    "page": 22,
                    "section": "Major Procedures",
                    "text": "Joint replacement is covered subject to medical necessity and pre-authorization.",
                },
                {
                    "source": "Sample Policy A",
                    "page": 7,
                    "section": "Exclusions",
                    "text": "Pre-existing conditions are excluded during the first policy year.",
                },
            ],
        }

    return {
        "query_id": "demo_qry_3",
        "answer": "Insufficient evidence from policy context to give a reliable yes/no decision.",
        "verdict": "Unclear",
        "confidence": 0.42,
        "reasoning": "The question needs a specific procedure name and policy details.",
        "explanation_local": "Sahi jawab ke liye specific procedure aur policy details chahiye.",
        "mode": "demo",
        "grounded": True,
        "policy_notice": "This assistant supports policy interpretation only and does not provide legal advice.",
        "citations": [
            {
                "source": "Sample Policy A",
                "page": 3,
                "section": "How to Read Coverage",
                "text": "Final admissibility depends on policy schedule, exclusions, and endorsements.",
            }
        ],
    }


def render_verdict_badge(verdict: str) -> None:
    verdict_clean = verdict.strip().lower()
    if verdict_clean == "yes":
        st.success("✅ **Verdict: COVERED**")
    elif verdict_clean == "no":
        st.error("❌ **Verdict: NOT COVERED**")
    elif verdict_clean == "supported":
        st.success("✅ **Verdict: SUPPORTED**")
    elif verdict_clean == "partially addressed":
        st.warning("⚠️ **Verdict: PARTIALLY ADDRESSED**")
    elif verdict_clean == "not mentioned":
        st.error("❌ **Verdict: NOT MENTIONED**")
    else:
        st.warning("⚠️ **Verdict: UNCLEAR - Verify with Insurer**")


def main() -> None:
    st.title("Health Insurance Policy Simplifier")
    st.caption("Upload policy PDFs and ask coverage questions with grounded citations.")

    with st.sidebar:
        st.subheader("Configuration")
        default_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        base_url = st.text_input("Backend URL", value=default_url).rstrip("/")
        demo_mode = st.toggle("Demo mode (no backend required)", value=False)

        if not demo_mode:
            healthy = _api_health_check(base_url)
            if healthy:
                st.success("Backend reachable")
            else:
                st.warning("Backend not reachable. You can still use demo mode.")

        st.markdown("---")
        st.caption("Safety: This is a decision-support assistant, not legal or claim approval advice.")

    if "current_policy_id" not in st.session_state:
        st.session_state.current_policy_id = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "policies" not in st.session_state:
        st.session_state.policies = {}
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "pending_question_input" not in st.session_state:
        st.session_state.pending_question_input = None

    if st.session_state.pending_question_input is not None:
        st.session_state.question_input = st.session_state.pending_question_input
        st.session_state.pending_question_input = None

    upload_col, info_col = st.columns([2, 1])
    with upload_col:
        st.subheader("1) Upload Policy")
        policy_name = st.text_input("Policy name", placeholder="e.g., Star Health Gold Plan")
        uploaded_file = st.file_uploader("Upload policy PDF", type=["pdf"])
        upload_clicked = st.button("Ingest Policy", use_container_width=True)

        if upload_clicked:
            if demo_mode:
                st.session_state.current_policy_id = "demo-policy-001"
                st.success("Demo policy loaded")
            elif uploaded_file is None:
                st.error("Please upload a PDF first")
            else:
                with st.spinner("Ingesting policy..."):
                    upload_result = upload_policy(base_url, uploaded_file, policy_name)
                if upload_result:
                    policy_id = upload_result["policy_id"]
                    st.session_state.current_policy_id = policy_id
                    st.session_state.policies[policy_id] = upload_result["policy_name"]
                    st.success(f"Policy ingested successfully. Policy ID: {policy_id}")

    with info_col:
        st.subheader("Current Session")
        st.write("Policy ID:", st.session_state.current_policy_id or "Not set")
        st.write("Mode:", "Demo" if demo_mode else "Live API")
        st.write("Loaded Policies:", len(st.session_state.policies))

    explain_language = st.selectbox("Explanation language", ["English", "Hindi", "Telugu"], index=0)
    compare_mode = st.toggle("Policy comparison mode (2 policies)", value=False)

    selected_compare_ids: List[str] = []
    if compare_mode and not demo_mode:
        policy_options = list(st.session_state.policies.keys())
        if len(policy_options) < 2:
            st.info("Upload at least 2 policies to use comparison mode.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                first_policy = st.selectbox(
                    "Policy A",
                    policy_options,
                    format_func=lambda pid: f"{st.session_state.policies.get(pid, pid)} ({pid})",
                    key="compare_policy_a",
                )
            with c2:
                second_candidates = [pid for pid in policy_options if pid != first_policy]
                second_policy = st.selectbox(
                    "Policy B",
                    second_candidates,
                    format_func=lambda pid: f"{st.session_state.policies.get(pid, pid)} ({pid})",
                    key="compare_policy_b",
                )
            selected_compare_ids = [first_policy, second_policy]

    st.markdown("---")

    st.subheader("2) Ask Coverage Questions")
    question = st.text_input(
        "Ask something",
        placeholder="Is knee replacement surgery covered after 2 years?",
        key="question_input"
    )
    
    st.caption("💡 Try these examples:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤰 Maternity coverage?", use_container_width=True):
            st.session_state.pending_question_input = "Is maternity hospitalization covered?"
            st.rerun()
    with col2:
        if st.button("🏥 Pre-existing conditions?", use_container_width=True):
            st.session_state.pending_question_input = "Are pre-existing diseases covered?"
            st.rerun()
    with col3:
        if st.button("⏰ Waiting period?", use_container_width=True):
            st.session_state.pending_question_input = "What is the waiting period for claims?"
            st.rerun()
    
    ask_clicked = st.button("Get Answer", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.error("Please enter a question")
        elif compare_mode and not demo_mode and len(selected_compare_ids) != 2:
            st.error("Please select 2 policies for comparison")
        elif not demo_mode and not st.session_state.current_policy_id and not compare_mode:
            st.error("Please ingest a policy before asking questions")
        else:
            with st.spinner("Finding answer from policy..."):
                if demo_mode:
                    result = demo_answer(question)
                elif compare_mode:
                    result = compare_question(base_url, question, selected_compare_ids, explain_language)
                else:
                    result = ask_question(base_url, question, st.session_state.current_policy_id, explain_language)
            st.session_state.history.insert(0, {"question": question, "result": result, "compare": compare_mode})

    if st.session_state.history:
        latest = st.session_state.history[0]["result"]
        st.markdown("---")
        st.subheader("Answer")
        if latest.get("comparison_mode"):
            if latest.get("error"):
                st.error(latest["error"])
            for idx, result in enumerate(latest.get("results", []), start=1):
                policy_name = "Policy"
                if idx - 1 < len(latest.get("policies", [])):
                    policy_name = latest["policies"][idx - 1].get("policy_name", policy_name)
                st.markdown(f"### {policy_name}")
                st.info(result.get("policy_notice", "This assistant supports policy interpretation only and does not provide legal advice."))
                st.caption(f"Question Type: {result.get('question_type', '-')}")
                render_verdict_badge(result.get("verdict", "UNCLEAR"))
                st.write(result.get("answer", "No answer"))
                if result.get("explanation_local"):
                    st.write("Plain-language:", result.get("explanation_local"))
                st.caption("Voice Output")
                _render_voice_player(result, explain_language)
                if result.get("confidence") is not None:
                    confidence_value = float(max(0.0, min(1.0, result["confidence"])))
                    st.progress(confidence_value, text=f"Confidence: {confidence_value:.0%}")
                if result.get("reasoning"):
                    with st.expander(f"Reasoning ({policy_name})"):
                        st.write(result["reasoning"])
        else:
            st.info(latest.get("policy_notice", "This assistant supports policy interpretation only and does not provide legal advice."))
            st.caption(f"Question Type: {latest.get('question_type', '-')}")
            render_verdict_badge(latest["verdict"])
            st.write(latest["answer"])

            if latest.get("explanation_local"):
                st.write("Plain-language:", latest.get("explanation_local"))

            st.caption("Voice Output")
            _render_voice_player(latest, explain_language)

            if latest.get("grounded"):
                st.success("Grounding: Document-supported response")
            else:
                st.warning("Grounding: Insufficient document support")

            if latest.get("confidence") is not None:
                confidence_value = float(max(0.0, min(1.0, latest["confidence"])))
                st.progress(confidence_value, text=f"Confidence: {confidence_value:.0%}")

            if latest.get("reasoning"):
                with st.expander("Reasoning"):
                    st.write(latest["reasoning"])

            st.subheader("Citations")
            citations = latest.get("citations", [])
            if not citations:
                st.info("No citations available")
            else:
                for idx, citation in enumerate(citations, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**{idx}. {citation['source']}** | Page: {citation['page']} | Section: {citation['section']}"
                        )
                        st.write(citation["text"])

    st.markdown("---")
    st.subheader("Evaluation Sheet")
    if demo_mode:
        st.info("Evaluation metrics are available in live API mode.")
    else:
        eval_data = get_evaluation(base_url)
        if not eval_data:
            st.warning("Could not fetch evaluation metrics from backend.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Queries", eval_data.get("total_queries", 0))
            with m2:
                acc = eval_data.get("accuracy")
                st.metric("Accuracy", f"{acc:.1%}" if isinstance(acc, (int, float)) else "N/A")
            with m3:
                cp = eval_data.get("citation_precision")
                st.metric("Citation Precision", f"{cp:.1%}" if isinstance(cp, (int, float)) else "N/A")
            with m4:
                fr = eval_data.get("fallback_rate", 0.0)
                st.metric("Fallback Rate", f"{fr:.1%}")

            if eval_data.get("reviewed_queries", 0) == 0:
                st.info("Accuracy is currently unavailable because no reviewed labels exist.")

            with st.expander("Recent Evaluation Rows"):
                rows = eval_data.get("recent_queries", [])
                if rows:
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.caption("No evaluation records yet")

    st.markdown("---")
    st.subheader("Claim Prediction (Simple)")
    if demo_mode:
        st.info("Claim prediction runs in live API mode.")
    else:
        colp1, colp2 = st.columns(2)
        with colp1:
            claim_policy_id = st.selectbox(
                "Policy for claim prediction",
                options=list(st.session_state.policies.keys()) or [""],
                format_func=lambda pid: st.session_state.policies.get(pid, "No policy loaded") if pid else "No policy loaded",
                key="claim_policy_selector",
            )
            claim_procedure = st.text_input("Procedure", placeholder="e.g., knee replacement surgery")
            claim_amount = st.number_input("Claim amount", min_value=0.0, value=150000.0, step=1000.0)
        with colp2:
            waiting_months = st.number_input("Waiting period completed (months)", min_value=0, value=12, step=1)
            has_ped = st.toggle("Has pre-existing condition", value=False)
            run_claim_prediction = st.button("Run Claim Prediction", use_container_width=True)

        if run_claim_prediction:
            if not claim_policy_id:
                st.error("Please upload and select a policy first")
            elif not claim_procedure.strip():
                st.error("Please enter a procedure")
            else:
                prediction = predict_claim(
                    base_url=base_url,
                    policy_id=claim_policy_id,
                    procedure=claim_procedure,
                    claim_amount=float(claim_amount),
                    waiting_period_completed_months=int(waiting_months),
                    has_pre_existing_condition=bool(has_ped),
                )
                if not prediction:
                    st.warning("Could not run claim prediction. Check backend status.")
                else:
                    st.metric("Approval Probability", f"{float(prediction.get('approval_probability', 0.0)):.1%}")
                    st.write("Risk Band:", prediction.get("risk_band", "-"))
                    factors = prediction.get("factors", [])
                    if factors:
                        st.write("Key Factors:")
                        for factor in factors:
                            st.write(f"- {factor}")

    with st.expander("Recent Questions"):
        if not st.session_state.history:
            st.caption("No questions asked yet")
        else:
            for item in st.session_state.history[:10]:
                st.write(f"• {item['question']}  →  {item['result']['verdict']}")


if __name__ == "__main__":
    main()
