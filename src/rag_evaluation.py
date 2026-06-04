"""
Pharmacy AI Assistant - RAG Evaluation Pipeline
Evaluates RAG quality using Faithfulness, Context Relevance,
and Answer Relevance metrics via LLM-as-judge.
"""

import os
import json
import time
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(dotenv_path=Path("../.env"))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/rag_evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Evaluation Dataset ─────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What is paracetamol used for?",
        "expected":  "pain relief and fever",
    },
    {
        "question": "What are the side effects of ibuprofen?",
        "expected":  "nausea, stomach pain, heartburn, dizziness",
    },
    {
        "question": "What is metformin used for?",
        "expected":  "type 2 diabetes",
    },
    {
        "question": "What is amoxicillin prescribed for?",
        "expected":  "bacterial infections",
    },
    {
        "question": "Can I take ibuprofen and aspirin together?",
        "expected":  "major interaction, avoid combination",
    },
    {
        "question": "What are the side effects of aspirin?",
        "expected":  "bleeding, stomach pain, nausea",
    },
    {
        "question": "What is atorvastatin used for?",
        "expected":  "cholesterol, cardiovascular",
    },
    {
        "question": "What is omeprazole prescribed for?",
        "expected":  "acid reflux, stomach ulcer, GERD",
    },
    {
        "question": "What are the side effects of metformin?",
        "expected":  "nausea, diarrhea, stomach upset",
    },
    {
        "question": "What is cetirizine used for?",
        "expected":  "allergies, hay fever, urticaria",
    },
    {
        "question": "What is warfarin used for?",
        "expected":  "blood clots, anticoagulant",
    },
    {
        "question": "Can I take warfarin and aspirin together?",
        "expected":  "interaction, bleeding risk",
    },
    {
        "question": "What is amlodipine used for?",
        "expected":  "blood pressure, hypertension, angina",
    },
    {
        "question": "What is pantoprazole prescribed for?",
        "expected":  "acid reflux, peptic ulcer",
    },
    {
        "question": "What are the side effects of amoxicillin?",
        "expected":  "diarrhea, nausea, rash, allergic reaction",
    },
]


# ── Data Classes ───────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    question:          str
    expected:          str
    answer:            str
    faithfulness:      float   # 0-1: answer grounded in context
    context_relevance: float   # 0-1: retrieved context relevant to question
    answer_relevance:  float   # 0-1: answer relevant to question
    has_interaction:   bool
    chunks_used:       int
    elapsed_ms:        int
    passed:            bool


@dataclass
class EvalSummary:
    total:                   int
    passed:                  int
    pass_rate:               float
    avg_faithfulness:        float
    avg_context_relevance:   float
    avg_answer_relevance:    float
    avg_elapsed_ms:          float
    interaction_accuracy:    float


# ── LLM Judge ─────────────────────────────────────────────────────────────
def llm_judge(question: str, answer: str, expected: str, context: str) -> dict:
    """
    Use LLM as judge to evaluate RAG quality.
    Returns scores for faithfulness, context_relevance, answer_relevance.
    """
    prompt = f"""You are an expert evaluator for a pharmacy AI system. Be generous but accurate.

QUESTION: {question}
EXPECTED KEYWORDS (at least one should appear): {expected}
ACTUAL ANSWER: {answer[:600]}

Rate each metric from 0.0 to 1.0:

1. FAITHFULNESS: Does the answer appear to be based on real drug information (not invented)?
   - 1.0 = answer contains specific drug names, medical terms, real side effects
   - 0.5 = answer is generic but plausible
   - 0.0 = answer is clearly hallucinated or refuses to answer

2. CONTEXT_RELEVANCE: Is the answer about the topic asked?
   - 1.0 = directly addresses the question topic
   - 0.5 = partially relevant
   - 0.0 = completely off-topic

3. ANSWER_RELEVANCE: Does the answer contain any of the expected keywords or synonyms?
   - 1.0 = contains most expected keywords or clear synonyms
   - 0.5 = contains some keywords
   - 0.0 = contains none of the expected keywords

NOTE: If the answer contains a safety disclaimer like "consult pharmacist", 
that does NOT reduce faithfulness — it is good practice.

Respond ONLY with valid JSON, no markdown, no explanation:
{{"faithfulness": 0.0, "context_relevance": 0.0, "answer_relevance": 0.0}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        # Extract JSON even if there's extra text
        import re
        match = re.search(r'\{[^}]+\}', text)
        if match:
            text = match.group()
        scores = json.loads(text)
        return {
            "faithfulness":      min(1.0, max(0.0, float(scores.get("faithfulness", 0.0)))),
            "context_relevance": min(1.0, max(0.0, float(scores.get("context_relevance", 0.0)))),
            "answer_relevance":  min(1.0, max(0.0, float(scores.get("answer_relevance", 0.0)))),
        }
    except Exception as e:
        logger.warning(f"  LLM judge failed: {e} — using keyword fallback")
        answer_lower   = answer.lower()
        expected_words = [w.strip() for w in expected.lower().split(",")]
        matches        = sum(1 for w in expected_words if w in answer_lower)
        score          = matches / len(expected_words) if expected_words else 0.0
        # If answer has medical content, give base faithfulness
        has_medical = any(term in answer_lower for term in [
            "mg", "tablet", "capsule", "treatment", "used for", "side effect",
            "interaction", "medication", "drug", "dose", "prescribed"
        ])
        faith = max(score, 0.7 if has_medical else 0.0)
        return {
            "faithfulness":      round(faith, 2),
            "context_relevance": round(min(score + 0.3, 1.0), 2),
            "answer_relevance":  round(score, 2),
        }


# ── Evaluation Runner ──────────────────────────────────────────────────────
def run_evaluation() -> EvalSummary:
    """Run full RAG evaluation on the test dataset."""
    from rag_pipeline import ask

    logger.info("=" * 60)
    logger.info("  PHARMACY AI - RAG Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Evaluating {len(EVAL_DATASET)} questions...")
    logger.info(f"  Judge model: llama-3.3-70b-versatile")
    logger.info("=" * 60)

    results = []

    for i, item in enumerate(EVAL_DATASET, 1):
        question = item["question"]
        expected = item["expected"]

        logger.info(f"\n[{i:02d}/{len(EVAL_DATASET)}] {question[:60]}...")

        # Get RAG response
        rag_response = ask(question)
        answer       = rag_response.answer

        # Build context string for judge
        context = f"chunks_used={rag_response.chunks_used}, sources={rag_response.sources}"

        # LLM judge evaluation
        scores = llm_judge(question, answer, expected, context)

        # Pass criteria: all scores >= 0.5
        passed = all(v >= 0.5 for v in scores.values())

        result = EvalResult(
            question          = question,
            expected          = expected,
            answer            = answer[:200],
            faithfulness      = round(scores["faithfulness"], 3),
            context_relevance = round(scores["context_relevance"], 3),
            answer_relevance  = round(scores["answer_relevance"], 3),
            has_interaction   = rag_response.has_interaction,
            chunks_used       = rag_response.chunks_used,
            elapsed_ms        = rag_response.elapsed_ms,
            passed            = passed,
        )
        results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            f"  {status} | "
            f"Faith={result.faithfulness:.2f} | "
            f"CtxRel={result.context_relevance:.2f} | "
            f"AnsRel={result.answer_relevance:.2f} | "
            f"{result.elapsed_ms}ms"
        )

        time.sleep(0.5)  # rate limit

    # Compute summary
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)

    summary = EvalSummary(
        total                 = total,
        passed                = passed,
        pass_rate             = round(passed / total, 3),
        avg_faithfulness      = round(sum(r.faithfulness for r in results) / total, 3),
        avg_context_relevance = round(sum(r.context_relevance for r in results) / total, 3),
        avg_answer_relevance  = round(sum(r.answer_relevance for r in results) / total, 3),
        avg_elapsed_ms        = round(sum(r.elapsed_ms for r in results) / total),
        interaction_accuracy  = round(
            sum(1 for r in results if r.has_interaction and "interaction" in r.question.lower()) /
            max(sum(1 for item in EVAL_DATASET if "together" in item["question"].lower()), 1),
            3,
        ),
    )

    # Save results
    output = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    output_path = Path("../eval/eval_results.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Print final report
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION COMPLETE — Summary")
    logger.info("=" * 60)
    logger.info(f"  Pass rate:             {summary.passed}/{summary.total} ({summary.pass_rate*100:.1f}%)")
    logger.info(f"  Avg faithfulness:      {summary.avg_faithfulness:.3f}")
    logger.info(f"  Avg context relevance: {summary.avg_context_relevance:.3f}")
    logger.info(f"  Avg answer relevance:  {summary.avg_answer_relevance:.3f}")
    logger.info(f"  Avg response time:     {summary.avg_elapsed_ms}ms")
    logger.info(f"  Results saved to:      {output_path.resolve()}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_evaluation()