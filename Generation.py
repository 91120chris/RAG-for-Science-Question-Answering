"""
HW3 Part 3: Generation with Local Mistral-7B-Instruct + Full Pipeline Evaluation
Course: RNN and Transformer, Spring 2026

Uses mistralai/Mistral-7B-Instruct-v0.3 loaded directly on RTX 4090
via HuggingFace transformers (same approach as HW2 LocalLLM.py).
No Ollama required.
"""

import re
import time
import gc
import warnings
import torch
import pandas as pd
from typing import Optional
from transformers import pipeline

warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`")

# ── Configuration ──────────────────────────────────────────────────────────────
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = (
    "You are a precise science exam assistant. "
    "Rules you MUST follow:\n"
    "1. Answer based ONLY on the provided context.\n"
    "2. If the answer is not in the context, respond: 'I don't know.'\n"
    "3. Output the letter (A/B/C/D/E) on the FIRST line, "
    "then a one-sentence explanation.\n"
    "4. Do NOT invent information absent from the context."
)

_generator = None   # loaded once, reused across calls


# ── Model loader ────────────────────────────────────────────────────────────────

def get_generator():
    """Load Mistral-7B-Instruct onto GPU (singleton – loaded only once)."""
    global _generator
    if _generator is None:
        print(f"Loading {GEN_MODEL_NAME} on GPU (fp16) ...")
        gc.collect()
        torch.cuda.empty_cache()
        _generator = pipeline(
            "text-generation",
            model=GEN_MODEL_NAME,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        print("Model loaded.\n")
    return _generator


# ── Prompt construction ─────────────────────────────────────────────────────────

def build_messages(question: str, options: dict[str, str], context: str) -> list[dict]:
    """
    Build a chat-template message list for Mistral Instruct.
    Mistral v0.3 supports system role via its chat template.
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    user_content  = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Which option is correct?  Reply with the letter first."
    )
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


def extract_answer_letter(response: str) -> Optional[str]:
    """
    Parse the LLM response and return the chosen letter (A–E), or None.
    Tries patterns from most to least strict.
    """
    if not response:
        return None

    patterns = [
        r"^([A-E])[.\):\s]",
        r"(?:answer|option|choice)\s+(?:is\s+)?([A-E])\b",
        r"^([A-E])$",
        r"\b([A-E])\b",
    ]
    for pat in patterns:
        m = re.search(pat, response.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


# ── RAG pipeline (single question) ─────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    options: dict[str, str],
    db,
    use_reranking: bool = True,
) -> dict:
    """
    Full RAG cycle for one question:
      1. Retrieve context (vector search ± re-ranking)
      2. Build messages
      3. Generate answer via local Mistral

    Returns a dict with: answer, response, context_docs, latency (breakdown).
    """
    from Retrieval import advanced_retrieve, vector_only_retrieve

    # ── Retrieval ───────────────────────────────────────────────────────────────
    if use_reranking:
        docs, _, v_lat, r_lat = advanced_retrieve(question, db)
    else:
        docs, v_lat = vector_only_retrieve(question, db, k=3)
        r_lat = 0.0

    # ── Generation ──────────────────────────────────────────────────────────────
    context_str = "\n\n---\n\n".join(d.page_content for d in docs)
    messages    = build_messages(question, options, context_str)

    gen = get_generator()

    t_gen = time.perf_counter()
    output = gen(
        messages,
        max_new_tokens=128,
        do_sample=False,         # greedy – deterministic answers
        temperature=None,
        top_p=None,
    )
    g_lat = time.perf_counter() - t_gen

    # Extract only the newly generated assistant turn
    raw_response = output[0]["generated_text"][-1]["content"]

    return {
        "answer":       extract_answer_letter(raw_response),
        "response":     raw_response,
        "context_docs": docs,
        "latency": {
            "vector_search": v_lat,
            "reranking":     r_lat,
            "generation":    g_lat,
            "total":         v_lat + r_lat + g_lat,
        },
    }


# ── Batch evaluation ────────────────────────────────────────────────────────────

def evaluate_pipeline(
    eval_df: pd.DataFrame,
    db,
    use_reranking: bool = True,
    verbose: bool       = True,
) -> dict:
    """
    Run the full RAG pipeline on every row in *eval_df* and collect
    accuracy + per-stage latency.

    Parameters
    ----------
    eval_df       : pre-sampled evaluation set (50 rows from split_dataset)
    db            : ChromaDB vector store to query
    use_reranking : whether to apply Cross-Encoder re-ranking
    verbose       : print per-question results
    """
    OPTION_COLS   = ["A", "B", "C", "D", "E"]
    correct_count = 0
    lats          = {"vector_search": [], "reranking": [], "generation": [], "total": []}
    no_answer     = []

    label = "With Re-ranking" if use_reranking else "Vector Only"
    print(f"\n{'='*60}")
    print(f"Evaluating [{label}] on {len(eval_df)} questions ...")
    print(f"{'='*60}")

    for enum_i, (_, row) in enumerate(eval_df.iterrows()):
        question = row["prompt"]
        options  = {col: row[col] for col in OPTION_COLS}
        truth    = row["answer"]

        result    = run_rag_pipeline(question, options, db, use_reranking)
        predicted = result["answer"]
        is_ok     = predicted == truth

        if is_ok:
            correct_count += 1
        if predicted is None:
            no_answer.append(enum_i)

        for k in lats:
            lats[k].append(result["latency"][k])

        if verbose:
            tag = "CORRECT" if is_ok else f"WRONG (pred={predicted}, truth={truth})"
            print(f"  Q{enum_i+1:3d}: {tag:35s}  total={result['latency']['total']:.2f}s")

    total = len(eval_df)
    acc   = correct_count / total if total else 0
    avg_l = {k: sum(v) / len(v) if v else 0 for k, v in lats.items()}

    print(f"\nResult: {correct_count}/{total} correct  ({acc:.2%})  |  "
          f"unanswered: {len(no_answer)}")

    return {
        "accuracy":      acc,
        "correct":       correct_count,
        "total":         total,
        "avg_latency":   avg_l,
        "no_answer":     no_answer,
        "use_reranking": use_reranking,
    }


# ── Final report ────────────────────────────────────────────────────────────────

def print_report(vec_res: dict, rer_res: dict) -> None:
    """Print the full evaluation report covering accuracy and latency."""
    print("\n" + "=" * 65)
    print("FINAL EVALUATION REPORT")
    print(f"Model: {GEN_MODEL_NAME}")
    print("=" * 65)

    print(f"\nAccuracy:")
    print(f"  Vector Search Only :  {vec_res['accuracy']:.2%}  "
          f"({vec_res['correct']}/{vec_res['total']})")
    print(f"  With Re-ranking    :  {rer_res['accuracy']:.2%}  "
          f"({rer_res['correct']}/{rer_res['total']})")

    print(f"\nAverage Latency per Query:")
    print(f"  {'Stage':<22} {'Vec Only':>10} {'Re-rank':>10}")
    print(f"  {'-'*44}")
    for label, key in [
        ("Vector Search",  "vector_search"),
        ("Re-ranking",     "reranking"),
        ("LLM Generation", "generation"),
        ("Total",          "total"),
    ]:
        v = vec_res['avg_latency'][key]
        r = rer_res['avg_latency'][key]
        print(f"  {label:<22} {v:>9.3f}s {r:>9.3f}s")

    acc_delta = rer_res['accuracy'] - vec_res['accuracy']
    lat_delta = rer_res['avg_latency']['total'] - vec_res['avg_latency']['total']

    print(f"\nConclusion:")
    if acc_delta > 0:
        print(f"  Re-ranking improved accuracy by {acc_delta:+.2%}.")
    elif acc_delta == 0:
        print(f"  Re-ranking did not change accuracy.")
    else:
        print(f"  Re-ranking changed accuracy by {acc_delta:.2%}.")
    print(f"  Re-ranking added {lat_delta:+.3f}s per query on average "
          f"({'worth it' if acc_delta > 0 else 'not worth it'} "
          f"given the accuracy trade-off).")


# ── Standalone entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from Chunking import split_dataset, load_csv_corpus, create_chunks_strategy_b
    from DB import get_embedding_model, build_index, CHROMA_DIR_B, COLLECTION_B

    print("=" * 60)
    print("Part 3: Generation with Mistral-7B-Instruct – Full Evaluation")
    print("=" * 60)

    eval_df, knowledge_df = split_dataset()
    raw_docs = load_csv_corpus(knowledge_df)
    docs_b   = create_chunks_strategy_b(raw_docs)

    embeddings = get_embedding_model()
    db_b = build_index(docs_b, embeddings, CHROMA_DIR_B, COLLECTION_B)

    vec_results = evaluate_pipeline(eval_df, db_b, use_reranking=False)
    rer_results = evaluate_pipeline(eval_df, db_b, use_reranking=True)

    print_report(vec_results, rer_results)
