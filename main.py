"""
HW3: RAG for Science Question Answering – Main Entry Point
Course: RNN and Transformer, Spring 2026

Data source: kaggle-llm-science-exam/train.csv only (no external downloads)
Evaluation : 50 questions randomly sampled from train.csv (seed=42)

Usage:
  python main.py                 # full pipeline (build + evaluate)
  python main.py --skip-build    # reuse saved ChromaDB indices
  python main.py --no-gen        # skip Ollama generation (indexing + retrieval only)
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

SKIP_BUILD = "--skip-build" in sys.argv
NO_GEN     = "--no-gen"     in sys.argv

print("=" * 65)
print("HW3: RAG for Science Question Answering")
print("Course: RNN and Transformer  |  Spring 2026")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1  Data Loading & Chunking
# ──────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1]  Data Loading & Chunking")
print("-" * 40)

from Chunking import (
    split_dataset,
    load_csv_corpus,
    create_chunks_strategy_a,
    create_chunks_strategy_b,
)

eval_df, knowledge_df = split_dataset()
print(f"\nTotal dataset : {len(eval_df) + len(knowledge_df)} questions")
print(f"  Eval set    : {len(eval_df)} questions  (seed=42, for generation evaluation)")
print(f"  Knowledge   : {len(knowledge_df)} questions  (RAG corpus)")

raw_docs = load_csv_corpus(knowledge_df)
n_docs   = len(raw_docs)

print("\nChunking Strategy Comparison:")
print("=" * 60)
docs_a = create_chunks_strategy_a(raw_docs)
docs_b = create_chunks_strategy_b(raw_docs)

print()
print(f"  {'Metric':<30} {'Method A':>12} {'Method B':>12}")
print(f"  {'-'*56}")
print(f"  {'Chunk Size':<30} {'500 tokens':>12} {'1000 tokens':>12}")
print(f"  {'Overlap':<30} {'50 tokens':>12} {'100 tokens':>12}")
print(f"  {'Overlap Ratio':<30} {'10%':>12} {'10%':>12}")
print(f"  {'Splitting Strategy':<30} {'Fixed-size':>12} {'Semantic/Rec.':>12}")
print(f"  {'Total Chunks':<30} {len(docs_a):>12} {len(docs_b):>12}")
print(f"  {'Avg Chunks / Document':<30} {len(docs_a)/n_docs:>11.1f} {len(docs_b)/n_docs:>11.1f}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2  Vector Database Construction
# ──────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2]  Building Vector Indices")
print("-" * 40)

from DB import (
    get_embedding_model, build_index,
    CHROMA_DIR_A, CHROMA_DIR_B,
    COLLECTION_A, COLLECTION_B,
)

embeddings = get_embedding_model()

print("\nIndex A – Fixed-size chunks (Strategy A) ...")
db_a = build_index(
    docs_a, embeddings, CHROMA_DIR_A, COLLECTION_A,
    force_rebuild=not SKIP_BUILD,
)

print("\nIndex B – Semantic chunks (Strategy B) ...")
db_b = build_index(
    docs_b, embeddings, CHROMA_DIR_B, COLLECTION_B,
    force_rebuild=not SKIP_BUILD,
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  Retrieval & Re-ranking
# ──────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3]  Retrieval & Re-ranking")
print("-" * 40)

from Retrieval import (
    vector_only_retrieve,
    advanced_retrieve,
    print_hit_rate_table,
    show_reranking_examples,
)

# Single-query demo with the first eval question
demo_q = eval_df["prompt"].iloc[0]
print(f"\nDemo query (eval Q1): {demo_q[:90]}...")

docs_v, lat_v = vector_only_retrieve(demo_q, db_b, k=3)
print(f"\n  [Vector Only]  latency={lat_v:.3f}s")
for i, d in enumerate(docs_v):
    print(f"    [{i+1}] {d.page_content[:80]}...")

advanced_retrieve(demo_q, db_b, verbose=True)

# Hit-rate comparison on 20 eval questions (subset for speed)
N_HR        = min(20, len(eval_df))
eval_sample = eval_df.head(N_HR)
questions   = eval_sample["prompt"].tolist()
correct_txt = [row[row["answer"]] for _, row in eval_sample.iterrows()]

print_hit_rate_table(questions, correct_txt, db_b)

# Re-ranking impact examples – search all 50 eval questions to maximise coverage
all_questions = eval_df["prompt"].tolist()
print("\n── Re-ranking Impact Examples (Report Section 2) ──")
show_reranking_examples(all_questions, eval_df, db_b, n_examples=2)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  Generation & Evaluation
# ──────────────────────────────────────────────────────────────────────────────
if NO_GEN:
    print("\n[STEP 4]  Skipped (--no-gen flag).")
else:
    print("\n[STEP 4]  Generation with Mistral-7B-Instruct – Full Evaluation")
    print("-" * 40)

    from Generation import evaluate_pipeline, print_report

    # Use Index B (semantic chunks give better context for generation)
    vec_results = evaluate_pipeline(eval_df, db_b, use_reranking=False, verbose=True)
    rer_results = evaluate_pipeline(eval_df, db_b, use_reranking=True,  verbose=True)

    print_report(vec_results, rer_results)

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Pipeline complete.")
print("=" * 65)
