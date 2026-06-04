"""
Pharmacy AI Assistant - RAG Pipeline
Retrieves relevant documents from Databricks Vector Search
and generates answers using Groq (Llama 3.3 70B).

Design principle: LLM decides everything — topic classification, drug extraction,
chunk relevance. No hardcoded drug lists, no keyword filters.
Retrieval decides what's known in the dataset.
"""

import os
import re
import time
import logging
import sys
import json
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from databricks.vector_search.client import VectorSearchClient

try:
    import mlflow
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False

load_dotenv(dotenv_path=Path("../.env"))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────
DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

VS_ENDPOINT      = os.getenv("DATABRICKS_VS_ENDPOINT", "pharmacy-ai-endpoint")
CATALOG          = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA           = os.getenv("DATABRICKS_SCHEMA", "pharmacy_ai")
INDEX_NAME       = f"{CATALOG}.{SCHEMA}.pharmacy_index"

EMBEDDING_MODEL  = "text-embedding-3-small"
EMBEDDING_DIM    = 1536
LLM_MODEL        = "llama-3.3-70b-versatile"
TOP_K            = 10
RERANK_TOP_K     = 6
MAX_TOKENS       = 1024
TEMPERATURE      = 0.1

DISCLAIMER = "⚠️ Always consult your pharmacist or physician before making medication decisions."

# Only truly safety-critical patterns stay as regex — too dangerous to wait for LLM
# Minimal regex patterns for IMMEDIATE safety blocks — only truly unambiguous self-harm intent
# Everything else is handled by llm_analyze_question which also checks for harmful intent
DANGEROUS_PATTERNS = [
    "kill myself", "suicide", "end my life", "how to die",
    "poison someone", "harm someone", "to harm myself", "to kill myself",
]

DOSAGE_PATTERNS = [
    "exact dose", "exact dosage", "how many mg", "how many milligrams",
    "maximum dose", "how much can i take", "can i take more",
]


# ── Clients ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)
vs_client     = VectorSearchClient(
    workspace_url=DATABRICKS_HOST,
    personal_access_token=DATABRICKS_TOKEN,
    disable_notice=True,
)


# ── Data Classes ───────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    doc_type:     str
    source:       str
    rag_text:     str
    score:        float
    rerank_score: float = 0.0


@dataclass
class RAGResponse:
    question:        str
    answer:          str
    chunks_used:     int
    sources:         list
    has_interaction: bool
    elapsed_ms:      int


# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional pharmacy assistant AI helping pharmacists and patients understand medications.

INSTRUCTIONS:
1. Answer the question using ONLY the provided context. Never invent information.
2. Be specific — mention drug names, side effects, interaction severity from the context.
3. For drug interactions, clearly state the severity level (major, moderate, minor).
4. If interaction data for a specific drug pair is missing from context, say:
   "I don't have sufficient data on [drug A] + [drug B] interaction — please consult a pharmacist."
   Do NOT assume the drug is safe. Absence of data does NOT mean absence of risk.
   Do NOT add information about other unrelated interactions as a substitute — stop there.
   The user asked about a specific combination; do not answer a different question.
5. Answer ONLY what the user asked. If the user asked about side effects, answer only side effects.
   Do NOT add interaction information for other drugs mentioned in patient reviews or context
   unless the user explicitly asked about interactions.
6. Never recommend one drug over another unless the context explicitly supports it.
7. Do NOT include meta-commentary about your reasoning process, what you are or are not doing,
   or why you chose not to discuss something. Just answer the question directly.
8. Always end with the safety disclaimer.

RESPONSE FORMAT:
- Direct, informative answer based on the context
- Specific details for each drug or interaction asked about
- End with: "⚠️ Always consult your pharmacist or physician before making medication decisions."
"""


# ── LLM Helpers ────────────────────────────────────────────────────────────
def llm_analyze_question(question: str) -> tuple[bool, bool, list[str]]:
    """
    Single LLM call that does both:
    1. Classifies if the question is pharmacy-related (YES/NO)
    2. Extracts drug/medication names from the question

    Combining both tasks into one call saves ~500ms latency per query.
    Returns (is_pharmacy: bool, drugs: list[str])
    """
    prompt = (
        "You are a pharmacy assistant classifier. Analyze the question and respond in JSON only.\n\n"
        "Return a JSON object with exactly these three fields:\n"
        "- \"is_pharmacy\": true if the question is about medications, drugs, interactions, "
        "dosage, or side effects; false otherwise\n"
        "- \"is_harmful\": true if the question asks about overdosing, dangerous doses, "
        "how many pills/tablets would be fatal or dangerous, lethal amounts, or any intent "
        "to harm oneself or others with medication; false otherwise\n"
        "- \"drugs\": array of lowercase drug/medication names mentioned in the question "
        "(empty array if none).\n\n"
        "Examples:\n"
        "Q: What is the weather today?\n"
        "→ {\"is_pharmacy\": false, \"is_harmful\": false, \"drugs\": []}\n\n"
        "Q: What is [drug name] used for?\n"
        "→ {\"is_pharmacy\": true, \"is_harmful\": false, \"drugs\": [\"drug name\"]}\n\n"
        "Q: How many [drug name] tablets would be dangerous?\n"
        "→ {\"is_pharmacy\": true, \"is_harmful\": true, \"drugs\": [\"drug name\"]}\n\n"
        "Q: Can I take [drug A] with [drug B]?\n"
        "→ {\"is_pharmacy\": true, \"is_harmful\": false, \"drugs\": [\"drug a\", \"drug b\"]}\n\n"
        f"Q: {question}\n"
        "→"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        is_pharmacy = bool(result.get("is_pharmacy", True))
        is_harmful  = bool(result.get("is_harmful", False))
        drugs = [d.lower().strip() for d in result.get("drugs", [])
                 if isinstance(d, str) and d.strip()]
        logger.info(f"  LLM analyze: is_pharmacy={is_pharmacy}, is_harmful={is_harmful}, drugs={drugs}")
        return is_pharmacy, is_harmful, drugs
    except Exception as e:
        logger.warning(f"  LLM analyze failed: {e} — defaulting to allow, no drugs")
        return True, False, []


# ── Input Guardrail ────────────────────────────────────────────────────────
def guardrail_input(question: str, is_pharmacy: bool = True, is_harmful: bool = False) -> tuple[bool, str]:
    """
    Block dangerous or off-topic questions.

    Step 1: Regex for dangerous patterns (fast, safety-critical)
    Step 2: Uses pre-computed LLM classification (is_pharmacy) — no extra LLM call
    """
    q_lower = question.lower().strip()

    off_topic_msg = (
        "I'm a pharmacy assistant and can only answer questions about medications, "
        "drug interactions, side effects, and related medical topics.\n\n"
        "Please ask me about a specific medication or drug interaction."
    )

    if len(question.strip()) < 5:
        return False, "Please provide a more detailed question about the medication."

    # Step 1 — Dangerous patterns (regex, no LLM needed)
    for pattern in DANGEROUS_PATTERNS:
        if pattern in q_lower:
            logger.warning(f"  GUARDRAIL: Dangerous pattern: {pattern}")
            return False, (
                "I can't provide instructions about overdosing on medication.\n\n"
                "🆘 If this relates to a real exposure or emergency:\n"
                "• Emergency services: 112\n"
                "• Poison Control Center: Contact your local poison center immediately\n\n"
                "If you have questions about safe medication use, I'm here to help."
            )

    # Step 2 — LLM harmful intent check (catches overdose/dangerous dose questions)
    if is_harmful:
        logger.warning("  GUARDRAIL: LLM detected harmful intent")
        return False, (
            "I can't provide information about dangerous or lethal doses of medication.\n\n"
            "🆘 If this relates to a real exposure or emergency:\n"
            "• Emergency services: 112\n"
            "• Poison Control Center: Contact your local poison center immediately\n\n"
            "If you have questions about safe medication use, I'm here to help."
        )

    # Step 3 — Use pre-computed LLM classification from ask()
    if not is_pharmacy:
        logger.warning("  GUARDRAIL: LLM classified as off-topic")
        return False, off_topic_msg

    return True, ""


# ── Embedding ──────────────────────────────────────────────────────────────
def embed_query(query: str) -> list:
    """Embed a query using OpenAI text-embedding-3-small."""
    response = openai_client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


# ── Retrieval ──────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, top_k: int = TOP_K) -> list:
    """Retrieve relevant chunks from Databricks Vector Search."""
    query_embedding = embed_query(query)
    try:
        index   = vs_client.get_index(VS_ENDPOINT, INDEX_NAME)
        results = index.similarity_search(
            query_vector=query_embedding,
            columns=["id", "doc_type", "source", "rag_text"],
            num_results=top_k,
        )
        chunks = []
        for item in results.get("result", {}).get("data_array", []):
            chunks.append(RetrievedChunk(
                doc_type = item[1] if len(item) > 1 else "unknown",
                source   = item[2] if len(item) > 2 else "unknown",
                rag_text = item[3] if len(item) > 3 else "",
                score    = float(item[4]) if len(item) > 4 else 0.0,
            ))
        logger.info(f"  Retrieved {len(chunks)} chunks for: '{query[:50]}'")
        return chunks
    except Exception as e:
        logger.error(f"  Retrieval failed: {e}")
        return []


# ── Multi-Query Retrieval ──────────────────────────────────────────────────
def retrieve_chunks_multi(question: str, drugs: list[str], top_k_per_pair: int = 5) -> list:
    """
    For multi-drug queries, run a separate retrieval per drug pair.
    Drug names come from LLM extraction — no hardcoded lists.

    Example: drugs = ["metformin", "warfarin", "aspirin"]
      → Search 1: "metformin warfarin interaction"
      → Search 2: "metformin aspirin interaction"
      → Search 3: "warfarin aspirin interaction"
      → Search 4: original question (general context)

    Capped at 3 pairs max to keep latency acceptable.
    """
    all_chunks = []
    seen_texts = set()

    if len(drugs) >= 2:
        pairs = list(combinations(drugs, 2))[:3]
        for drug_a, drug_b in pairs:
            # Search both orders: "warfarin aspirin" AND "aspirin warfarin"
            # This improves recall for asymmetric dataset entries
            for pair_query in [
                f"{drug_a} {drug_b} interaction",
                f"{drug_b} {drug_a} interaction",
            ]:
                logger.info(f"  Multi-query: '{pair_query}'")
                for chunk in retrieve_chunks(pair_query, top_k=top_k_per_pair):
                    if chunk.rag_text not in seen_texts:
                        seen_texts.add(chunk.rag_text)
                        all_chunks.append(chunk)

    # Always run original question too for general drug info
    for chunk in retrieve_chunks(question, top_k=TOP_K if len(drugs) < 2 else 4):
        if chunk.rag_text not in seen_texts:
            seen_texts.add(chunk.rag_text)
            all_chunks.append(chunk)

    logger.info(f"  Multi-query total: {len(all_chunks)} unique chunks")
    return all_chunks


# ── Reranking ──────────────────────────────────────────────────────────────
def rerank_chunks(query: str, chunks: list, drugs: list[str], top_k: int = RERANK_TOP_K) -> list:
    """
    Rerank chunks by relevance with guaranteed coverage per drug pair.

    Problem solved: for 3-drug queries (A+B+C), reranking by score alone
    drops pair chunks (e.g. B+C) that score lower than dominant pairs (A+B).

    Solution: after scoring, guarantee at least 1 chunk per drug pair
    before applying the top_k cap.
    """
    if not chunks:
        return []

    stopwords = {"what", "is", "are", "the", "a", "an", "i", "can", "take", "for", "of", "with"}
    query_terms = set(query.lower().split()) - stopwords

    for chunk in chunks:
        text_lower = chunk.rag_text.lower()
        score = chunk.score

        term_matches = sum(1 for t in query_terms if t in text_lower)
        score += term_matches * 0.05

        if any(w in query.lower() for w in ["together", "interact", "combination", "combine", "also take"]):
            if chunk.doc_type == "drug_interaction":
                score += 0.2

        if any(w in query.lower() for w in ["used for", "prescribed", "treat"]):
            if chunk.doc_type == "medicine_info":
                score += 0.1

        # Boost chunk that mentions ANY pair of asked drugs together
        if drugs:
            pairs_in_chunk = sum(
                1 for d in drugs if d in text_lower
            )
            if pairs_in_chunk >= 2:
                score += 0.15 * pairs_in_chunk  # more drugs mentioned = higher boost

        chunk.rerank_score = round(score, 4)

    reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    # Guarantee at least 1 chunk per drug pair (prevents pair starvation)
    if len(drugs) >= 2:
        guaranteed = []
        guaranteed_texts = set()
        pairs = list(combinations(drugs, 2))

        for drug_a, drug_b in pairs:
            for chunk in reranked:
                text_lower = chunk.rag_text.lower()
                if (drug_a in text_lower and drug_b in text_lower
                        and chunk.rag_text not in guaranteed_texts):
                    guaranteed.append(chunk)
                    guaranteed_texts.add(chunk.rag_text)
                    break  # 1 per pair is enough

        # Fill remaining slots with highest-scored chunks not already included
        remaining_slots = max(top_k, len(guaranteed) + 2)
        for chunk in reranked:
            if chunk.rag_text not in guaranteed_texts:
                guaranteed.append(chunk)
                guaranteed_texts.add(chunk.rag_text)
            if len(guaranteed) >= remaining_slots:
                break

        logger.info(f"  Reranked → {len(guaranteed)} chunks "
                    f"({len(pairs)} pairs guaranteed, drugs: {drugs})")
        return guaranteed

    # Single drug — standard top_k
    logger.info(f"  Reranked → top {top_k} chunks")
    return reranked[:top_k]


# ── Chunk Filter ───────────────────────────────────────────────────────────
def filter_chunks_by_drugs(chunks: list, drugs: list[str]) -> list:
    """
    Remove chunks that don't mention any of the drugs asked about.
    Only runs when LLM has identified specific drug names.

    This prevents off-drug chunks (e.g. Ascorbic acid when asking about
    Aspirin/Warfarin) from polluting the context.

    Safety net: if all chunks are filtered out, return originals.
    """
    if not drugs:
        return chunks  # no specific drugs — don't filter

    filtered = []
    for chunk in chunks:
        text_lower = chunk.rag_text.lower()
        if any(drug in text_lower for drug in drugs):
            filtered.append(chunk)
        else:
            logger.info(f"  FILTER: Removed chunk not mentioning {drugs}: "
                        f"'{chunk.rag_text[:50].strip()}'...")

    return filtered if filtered else chunks  # safety net


# ── Relevance Check ────────────────────────────────────────────────────────
def chunks_are_relevant(question: str, drugs: list[str], chunks: list) -> bool:
    """
    Check if retrieved chunks contain information about the asked drugs.

    Two cases:
    A) LLM extracted drug names → check if any chunk mentions them
    B) LLM extracted no drug names but question looks like a specific drug query
       → extract candidate words from question and check chunks
    """
    if not chunks:
        return False

    all_text = " ".join(c.rag_text.lower() for c in chunks)

    # Case A: LLM found specific drug names
    if drugs:
        matched = [d for d in drugs if d in all_text]
        if not matched:
            logger.info(f"  RELEVANCE: None of {drugs} found in chunks — not in DB")
            return False
        logger.info(f"  RELEVANCE: Matched: {matched}")
        return True

    # Case B: LLM found no drug names — extract candidates from question
    # If the question contains a specific noun that doesn't appear in any chunk,
    # it's likely an unknown medication
    skip = {"what", "is", "are", "the", "a", "an", "i", "can", "for", "of",
            "with", "about", "side", "effects", "dosage", "dose", "used",
            "interactions", "interaction", "recommended", "take", "taking"}
    words = [w.lower() for w in question.replace("?","").split()
             if len(w) > 3 and w.lower() not in skip]

    if words:
        matched = [w for w in words if w in all_text]
        if not matched:
            logger.info(f"  RELEVANCE: Candidates {words} not in chunks — unknown drug")
            return False

    return True


# ── Context Builder ────────────────────────────────────────────────────────
def build_context(chunks: list) -> tuple[str, bool]:
    """Build context string and detect drug interactions."""
    if not chunks:
        return "No relevant information found.", False

    has_interaction = any(c.doc_type == "drug_interaction" for c in chunks)
    has_major = any(
        "major" in c.rag_text.lower()
        for c in chunks if c.doc_type == "drug_interaction"
    )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = {
            "medicine_info":    "Medicine Information",
            "drug_interaction": "Drug Interaction Alert",
            "patient_review":   "Patient Experience",
        }.get(chunk.doc_type, "Medical Information")
        context_parts.append(f"[Source {i} — {label}]\n{chunk.rag_text}")

    context = "\n\n".join(context_parts)
    if has_major:
        context = "⚠️ MAJOR DRUG INTERACTION DETECTED IN CONTEXT ⚠️\n\n" + context

    return context, has_interaction


# ── Generation ─────────────────────────────────────────────────────────────
def generate_answer(question: str, context: str, drugs: list[str]) -> str:
    """Generate answer using Groq LLM with retrieved context."""

    # If specific drugs were asked, add focus constraint
    focus = ""
    if drugs:
        focus = (
            f"\nFOCUS: The user asked specifically about: {', '.join(drugs)}. "
            f"Address each drug mentioned. For pairs not found in context, "
            f"say 'I don't have sufficient data on [X] + [Y] — consult a pharmacist.'\n"
        )

    user_message = f"""DRUG INFORMATION CONTEXT:
{context}

PATIENT/PHARMACIST QUESTION:
{question}
{focus}
Answer based ONLY on the context above. Do not invent information."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


# ── Output Guardrail ───────────────────────────────────────────────────────
def guardrail_output(question: str, answer: str, chunks: list, drugs: list[str]) -> str:
    """Validate and enhance the generated answer."""
    q_lower = question.lower()

    off_topic_msg = (
        "I'm a pharmacy assistant and can only answer questions about medications, "
        "drug interactions, side effects, and related medical topics.\n\n"
        "Please ask me about a specific medication or drug interaction."
    )

    # No chunks at all
    if not chunks:
        logger.warning("  GUARDRAIL: No chunks retrieved")
        return off_topic_msg

    # Drugs asked but none found in DB
    if not chunks_are_relevant(question, drugs, chunks):
        logger.warning("  GUARDRAIL: Drugs not found in DB")
        return off_topic_msg

    # Low confidence
    scores = [c.rerank_score for c in chunks if hasattr(c, "rerank_score") and c.rerank_score]
    if scores and max(scores) < 0.3:
        logger.warning(f"  GUARDRAIL: Low confidence: {max(scores):.3f}")
        answer = (
            "⚠️ Note: The information below may not be directly relevant. "
            "Please verify with a healthcare professional.\n\n"
        ) + answer

    # Add disclaimer if missing
    OUT_OF_SCOPE_PHRASES = [
        "outside the scope", "does not contain information",
        "context does not contain", "provided context does not",
    ]
    is_out_of_scope = any(p in answer.lower() for p in OUT_OF_SCOPE_PHRASES)
    if not is_out_of_scope and DISCLAIMER not in answer:
        answer = answer + f"\n\n{DISCLAIMER}"

    # Major interaction warning banner
    major_keywords = ["major interaction", "avoid combination", "serious adverse", "major — avoid"]
    if any(kw in answer.lower() for kw in major_keywords):
        warning = "🚨 MAJOR INTERACTION WARNING: Do not combine these medications without physician approval."
        if warning not in answer:
            answer = f"{warning}\n\n{answer}"
        logger.warning("  GUARDRAIL: Major interaction detected")

    # Dosage disclaimer
    if any(p in q_lower for p in DOSAGE_PATTERNS):
        note = (
            "\n\n📋 DOSAGE NOTE: Specific dosages must be determined by a licensed physician "
            "based on your individual health condition, weight, age, and other medications."
        )
        if note not in answer:
            answer = answer + note

    # Medical advice disclaimer
    advice_phrases = ["you should take", "take this medication",
                      "i recommend taking", "the correct dose is", "take exactly"]
    if any(p in answer.lower() for p in advice_phrases):
        answer = answer + (
            "\n\n⚠️ NOTE: This information is for educational purposes only "
            "and does not replace professional medical advice."
        )

    return answer


# ── Main RAG Function ──────────────────────────────────────────────────────
def ask(question: str) -> RAGResponse:
    """
    Main RAG pipeline — fully LLM-driven

    Pipeline:
        1. Input guardrail (dangerous patterns + LLM topic classifier)
        2. LLM drug extraction (what drugs are in this question?)
        3. Multi-query retrieval (one search per drug pair)
        4. Reranking (relevance boosting, drug-aware)
        5. Chunk filter (remove chunks about wrong drugs)
        6. Context building
        7. LLM answer generation (with drug focus constraint)
        8. Output guardrail (relevance check, disclaimers, warnings)
    """
    start = time.time()
    logger.info(f"Processing: {question[:80]}...")

    # Step 1 — Single LLM call: classify topic + harmful intent + extract drugs
    is_pharmacy, is_harmful, drugs = llm_analyze_question(question)

    # Step 1b — Input guardrail (uses pre-computed classification)
    is_safe, rejection_msg = guardrail_input(question, is_pharmacy=is_pharmacy, is_harmful=is_harmful)
    if not is_safe:
        elapsed_ms = int((time.time() - start) * 1000)
        return RAGResponse(
            question=question, answer=rejection_msg,
            chunks_used=0, sources=[], has_interaction=False, elapsed_ms=elapsed_ms,
        )

    # Step 3 — Multi-query retrieval
    chunks = retrieve_chunks_multi(question, drugs)

    # Step 4 — Rerank (drug-aware)
    chunks = rerank_chunks(question, chunks, drugs)

    # Step 5 — Filter chunks to only those mentioning asked drugs
    chunks = filter_chunks_by_drugs(chunks, drugs)

    # Step 6 — Build context
    context, has_interaction = build_context(chunks)

    # Step 7 — Generate answer
    raw_answer = generate_answer(question, context, drugs) if chunks else ""

    # Step 8 — Output guardrail
    final_answer = guardrail_output(question, raw_answer, chunks, drugs)

    # If guardrail rejected (off-topic/not found), clear chunks for clean response
    off_topic_msg = "I'm a pharmacy assistant and can only answer questions"
    if off_topic_msg in final_answer:
        chunks = []

    elapsed_ms = int((time.time() - start) * 1000)

    sources = []
    seen = set()
    for c in chunks:
        first_line = c.rag_text.split("\n")[0] if c.rag_text else ""
        name = (first_line
            .replace("MEDICINE:", "").replace("DRUG INTERACTION ALERT", "Drug Interaction")
            .replace("FDA DRUG LABEL —", "").replace("PATIENT REVIEW -", "").strip()
        )
        if name and name not in seen:
            seen.add(name)
            sources.append(name)

    logger.info(f"  Done in {elapsed_ms}ms | drugs={drugs} | chunks={len(chunks)}")

    response = RAGResponse(
        question=question, answer=final_answer,
        chunks_used=len(chunks), sources=sources,
        has_interaction=has_interaction, elapsed_ms=elapsed_ms,
    )

    if MLFLOW_ENABLED:
        try:
            with mlflow.start_run(run_name="rag_query", nested=True):
                mlflow.log_param("question",        question[:200])
                mlflow.log_param("llm_model",       LLM_MODEL)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL)
                mlflow.log_param("drugs_detected",  str(drugs))
                mlflow.log_param("top_k",           TOP_K)
                mlflow.log_param("rerank_top_k",    RERANK_TOP_K)
                mlflow.log_metric("chunks_retrieved", len(chunks))
                mlflow.log_metric("elapsed_ms",       elapsed_ms)
                mlflow.log_metric("has_interaction",  int(has_interaction))
                mlflow.log_metric("answer_length",    len(final_answer))
        except Exception:
            pass

    return response


# ── CLI Test ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PHARMACY AI - RAG Pipeline Test")
    print("=" * 60)

    test_questions = [
        "What are the side effects of ibuprofen?",
        "Can I take ibuprofen and aspirin together?",
        "What is metformin used for?",
        "I take Metformin and Warfarin. Can I also take Aspirin?",
        "What is XyzMed123 used for?",
        "What is the capital of Kosovo?",
        "What is the weather today?",
        "How to overdose on paracetamol?",
        "What is the exact dose of aspirin?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {question}")
        print("─" * 60)
        response = ask(question)
        print(f"Answer:\n{response.answer}")
        print(f"\n  Chunks: {response.chunks_used} | "
              f"Interaction: {response.has_interaction} | "
              f"Elapsed: {response.elapsed_ms}ms")


if __name__ == "__main__":
    main()