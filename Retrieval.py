"""
HW3 Part 2: Retrieval & Re-ranking System
Course: RNN and Transformer, Spring 2026

Two-stage retrieval pipeline:
  Stage 1 – Dense vector search      (fast, high recall)
  Stage 2 – Cross-Encoder re-ranking (slow, high precision)

Also provides hit-rate comparison between the two approaches.
"""

import time
from typing import Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ── Configuration ──────────────────────────────────────────────────────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K    = 20    # Stage 1: vector candidates
FINAL_K      = 3     # Stage 2: after re-ranking

_reranker: Optional[CrossEncoder] = None   # loaded once


# ── Model loader ───────────────────────────────────────────────────────────────

def get_reranker(model_name: str = RERANK_MODEL) -> CrossEncoder:
    """Return (and cache) the Cross-Encoder model on CUDA."""
    global _reranker
    if _reranker is None:
        print(f"Loading Cross-Encoder: {model_name}  [device=cuda]")
        _reranker = CrossEncoder(model_name, device="cuda")
    return _reranker


# ── Retrieval functions ─────────────────────────────────────────────────────────

def vector_only_retrieve(
    query: str,
    db,
    k: int = FINAL_K,
) -> tuple[list[Document], float]:
    """
    Stage 1 only: dense vector similarity search.

    Returns
    -------
    docs    : top-k Documents
    latency : wall-clock seconds for the search
    """
    t0   = time.perf_counter()
    docs = db.similarity_search(query, k=k)
    return docs, time.perf_counter() - t0


def advanced_retrieve(
    query: str,
    db,
    initial_k: int = INITIAL_K,
    final_k: int   = FINAL_K,
    verbose: bool  = False,
) -> tuple[list[Document], list[float], float, float]:
    """
    Two-stage retrieval: vector search → Cross-Encoder re-ranking.

    Returns
    -------
    final_docs    : top-final_k Documents after re-ranking
    scores        : Cross-Encoder scores for each returned doc
    vector_latency: time for Stage 1 (seconds)
    rerank_latency: time for Stage 2 (seconds)
    """
    reranker = get_reranker()

    # Stage 1 ──────────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    candidates = db.similarity_search(query, k=initial_k)
    v_lat = time.perf_counter() - t1

    if not candidates:
        return [], [], v_lat, 0.0

    # Stage 2 ──────────────────────────────────────────────────────────────────
    t2    = time.perf_counter()
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    r_lat    = time.perf_counter() - t2

    final_docs    = [doc for _, doc in ranked[:final_k]]
    final_scores  = [float(s) for s, _ in ranked[:final_k]]

    if verbose:
        print(f"\n  Query: {query[:80]}")
        print(f"  Stage 1 – vector search   : {len(candidates)} candidates "
              f"({v_lat:.3f}s)")
        print(f"  Stage 2 – re-ranking      : top-{final_k} selected "
              f"({r_lat:.3f}s)")
        for i, (s, doc) in enumerate(ranked[:final_k]):
            print(f"    [{i+1}] score={s:.4f} | {doc.page_content[:70]}...")

    return final_docs, final_scores, v_lat, r_lat


# ── Hit-rate evaluation ─────────────────────────────────────────────────────────

def compute_hit_rate(
    questions: list[str],
    correct_answers: list[str],
    db,
    k: int          = FINAL_K,
    use_reranking: bool = True,
) -> dict:
    """
    Estimate retrieval quality via keyword-overlap hit rate.

    A question is considered a "hit" if any of the top-k retrieved chunks
    shares ≥ 30 % of the content words with the correct answer text.

    Returns a dict with: hit_rate, hits, total, avg_vector_latency,
    avg_rerank_latency.
    """
    hits = 0
    v_lats, r_lats = [], []

    for query, correct in zip(questions, correct_answers):
        if use_reranking:
            docs, _, v_lat, r_lat = advanced_retrieve(query, db, final_k=k)
            r_lats.append(r_lat)
        else:
            docs, v_lat = vector_only_retrieve(query, db, k=k)
        v_lats.append(v_lat)

        correct_words = set(correct.lower().split())
        hit = any(
            len(correct_words & set(d.page_content.lower().split()))
            / max(len(correct_words), 1) >= 0.30
            for d in docs
        )
        if hit:
            hits += 1

    n = len(questions)
    return {
        "hit_rate":           hits / n if n else 0,
        "hits":               hits,
        "total":              n,
        "avg_vector_latency": sum(v_lats) / n if n else 0,
        "avg_rerank_latency": sum(r_lats) / n if n else 0,
    }


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "it", "its", "that", "this",
    "and", "or", "but", "not", "if", "so", "than", "then", "when",
    "which", "who", "what", "where", "how", "all", "any", "each",
}


def _contains_answer(doc_content: str, answer_text: str, threshold: float = 0.30) -> bool:
    """
    Return True if doc_content shares >= threshold of answer_text's
    non-stop-word content words.
    """
    def keywords(text):
        return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

    answer_kw  = keywords(answer_text)
    content_kw = keywords(doc_content)
    if not answer_kw:
        return False
    return len(answer_kw & content_kw) / len(answer_kw) >= threshold


def _query_relevance(doc_content: str, query: str) -> float:
    """Keyword overlap between doc and query (excluding stop words)."""
    def keywords(text):
        return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

    q_kw = keywords(query)
    d_kw = keywords(doc_content)
    if not q_kw:
        return 0.0
    return len(q_kw & d_kw) / len(q_kw)


def show_reranking_examples(
    questions: list[str],
    df,
    db,
    n_examples: int = 2,
):
    """
    Print n_examples cases where:
      1. Vector Search top-1 was irrelevant to the query (low query keyword overlap)
      2. Re-ranking promoted a more relevant document to top-1 (higher query overlap)

    Qualifying condition: rerank_top1_relevance > vec_top1_relevance AND the new
    top-1 has >= 30% query keyword overlap (i.e., it is actually query-relevant).
    """
    reranker = get_reranker()
    all_cases = []

    for question in questions:
        row_match = df[df["prompt"] == question]
        if row_match.empty:
            continue
        answer_letter = row_match.iloc[0]["answer"]
        answer_text   = row_match.iloc[0][answer_letter]
        answer_line   = f"{answer_letter}. {answer_text}"

        candidates = db.similarity_search(question, k=20)
        if not candidates:
            continue

        pairs  = [[question, d.page_content] for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        vec_top1    = candidates[0]
        rerank_top1 = ranked[0][1]
        rerank_top1_score = ranked[0][0]

        # Skip if top-1 didn't change
        if vec_top1.page_content == rerank_top1.page_content:
            continue

        vec_rel    = _query_relevance(vec_top1.page_content,    question)
        rerank_rel = _query_relevance(rerank_top1.page_content, question)

        # Qualifying: reranker chose a MORE relevant doc AND it's genuinely relevant
        if rerank_rel <= vec_rel:
            continue
        if rerank_rel < 0.25:
            continue

        improvement = rerank_rel - vec_rel
        all_cases.append({
            "question":           question,
            "answer_line":        answer_line,
            "vec_top1":           vec_top1,
            "rerank_top1":        rerank_top1,
            "rerank_top1_score":  rerank_top1_score,
            "vec_rel":            vec_rel,
            "rerank_rel":         rerank_rel,
            "improvement":        improvement,
        })

    # Show the 2 cases with the largest relevance improvement
    all_cases.sort(key=lambda x: x["improvement"], reverse=True)
    top_cases = all_cases[:n_examples]

    if not top_cases:
        print("\n[Note] No qualifying examples found in the searched questions.")
        print(f"{'='*70}")
        return

    for i, c in enumerate(top_cases):
        print(f"\n{'='*70}")
        print(f"[Example {i+1}]")
        print(f"\nQuery:\n  {c['question']}")
        print(f"\nCorrect Answer:\n  {c['answer_line']}")
        print(f"\nBEFORE Re-ranking — Vector Search Top-1 "
              f"(query relevance: {c['vec_rel']:.0%}):")
        print(f"  {c['vec_top1'].page_content}")
        print(f"\nAFTER Re-ranking — Cross-Encoder Top-1 "
              f"(score={c['rerank_top1_score']:.4f}, query relevance: {c['rerank_rel']:.0%}):")
        print(f"  {c['rerank_top1'].page_content}")

    print(f"\n{'='*70}")


def print_hit_rate_table(
    questions: list[str],
    correct_answers: list[str],
    db,
    k: int = FINAL_K,
) -> None:
    """
    Compute and print the Hit Rate comparison table for the report.
    Compares 'Vector Search Only' vs 'Vector Search + Re-ranking'.
    """
    print(f"\n{'='*70}")
    print(f"Hit Rate Comparison: Vector Search Only vs. Vector Search + Re-ranking")
    print(f"(N={len(questions)} questions, top-k={k})")
    print(f"{'='*70}")

    print("  Computing Vector-Only hit rate ...")
    vec_hr = compute_hit_rate(questions, correct_answers, db, k=k, use_reranking=False)

    print("  Computing Re-ranking hit rate ...")
    rer_hr = compute_hit_rate(questions, correct_answers, db, k=k, use_reranking=True)

    print(f"\n  {'Approach':<26} {'Hit Rate':>10} {'Hits':>8}  "
          f"{'Avg Vec (s)':>12}  {'Avg Rerank (s)':>14}")
    print(f"  {'-'*74}")
    print(f"  {'Vector Search Only':<26} {vec_hr['hit_rate']:>9.2%} "
          f"{vec_hr['hits']:>5}/{vec_hr['total']:<3}  "
          f"{vec_hr['avg_vector_latency']:>11.3f}  {'N/A':>14}")
    print(f"  {'Vector Search + Re-ranking':<26} {rer_hr['hit_rate']:>9.2%} "
          f"{rer_hr['hits']:>5}/{rer_hr['total']:<3}  "
          f"{rer_hr['avg_vector_latency']:>11.3f}  "
          f"{rer_hr['avg_rerank_latency']:>13.3f}")

    delta = rer_hr['hit_rate'] - vec_hr['hit_rate']
    print(f"\n  Hit Rate delta: {delta:+.2%} from re-ranking")
    print(f"{'='*70}")

    return vec_hr, rer_hr


# ── Standalone demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from Chunking import (
        load_questions_df, load_wikipedia_corpus,
        create_chunks_strategy_a, create_chunks_strategy_b,
    )
    from DB import (
        get_embedding_model, build_index,
        CHROMA_DIR_A, CHROMA_DIR_B, COLLECTION_A, COLLECTION_B,
    )

    print("=" * 60)
    print("Part 2: Retrieval & Re-ranking")
    print("=" * 60)

    df        = load_questions_df()
    raw_docs  = load_wikipedia_corpus(df["prompt"].tolist())
    docs_a    = create_chunks_strategy_a(raw_docs)
    docs_b    = create_chunks_strategy_b(raw_docs)

    embeddings = get_embedding_model()
    db_a = build_index(docs_a, embeddings, CHROMA_DIR_A, COLLECTION_A)
    db_b = build_index(docs_b, embeddings, CHROMA_DIR_B, COLLECTION_B)

    # ── Demo: single query ──────────────────────────────────────────────────────
    test_q = df["prompt"].iloc[0]
    print(f"\nDemo query: {test_q[:90]}...\n")

    docs_v, lat_v = vector_only_retrieve(test_q, db_b, k=3)
    print(f"[Vector Only]  latency={lat_v:.3f}s")
    for i, d in enumerate(docs_v):
        print(f"  [{i+1}] {d.page_content[:80]}...")

    advanced_retrieve(test_q, db_b, verbose=True)

    # ── Hit-rate comparison (on 20 questions for speed) ─────────────────────────
    N = min(20, len(df))
    questions   = df["prompt"].tolist()[:N]
    option_cols = ["A", "B", "C", "D", "E"]
    correct_txt = [row[row["answer"]] for _, row in df.head(N).iterrows()]

    print(f"\n{'='*40}")
    print(f"Hit-Rate Comparison (N={N}, Index B):")
    print(f"{'='*40}")

    print("Computing Vector-Only hit rate ...")
    vec_hr = compute_hit_rate(questions, correct_txt, db_b, use_reranking=False)

    print("Computing Re-ranked hit rate ...")
    rer_hr = compute_hit_rate(questions, correct_txt, db_b, use_reranking=True)

    print(f"\n  Vector Only    : {vec_hr['hit_rate']:.2%}  "
          f"({vec_hr['hits']}/{vec_hr['total']})  "
          f"avg_vec={vec_hr['avg_vector_latency']:.3f}s")
    print(f"  With Re-ranking: {rer_hr['hit_rate']:.2%}  "
          f"({rer_hr['hits']}/{rer_hr['total']})  "
          f"avg_vec={rer_hr['avg_vector_latency']:.3f}s  "
          f"avg_rerank={rer_hr['avg_rerank_latency']:.3f}s")

    # ── Report: re-ranking examples ─────────────────────────────────────────────
    print("\n── Re-ranking Impact Examples (for report) ──")
    show_reranking_examples(questions, df, db_b, n_examples=2)
