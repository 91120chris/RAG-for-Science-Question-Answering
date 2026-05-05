"""
HW3 Part 1 (continued): Vector Database Construction
Course: RNN and Transformer, Spring 2026

Builds two separate ChromaDB indices:
  - Index A  →  chroma_db_fixed    (Strategy A fixed-size chunks)
  - Index B  →  chroma_db_semantic (Strategy B semantic chunks)

Embedding model: BAAI/bge-m3 on CUDA (RTX 4090)
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ──────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"
CHROMA_DIR_A = "./chroma_db_fixed"    # Persisted index for Strategy A
CHROMA_DIR_B = "./chroma_db_semantic" # Persisted index for Strategy B
COLLECTION_A = "strategy_a_fixed"
COLLECTION_B = "strategy_b_semantic"


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_embedding_model(model_name: str = EMBED_MODEL) -> HuggingFaceEmbeddings:
    """Load the embedding model onto GPU (singleton-friendly)."""
    print(f"Loading embedding model: {model_name}  [device=cuda]")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    documents,
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str,
    collection_name: str,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Build a ChromaDB index from *documents* and persist it to *persist_dir*.
    If the directory already exists (and force_rebuild=False), the existing
    index is loaded instead of being rebuilt.
    """
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"Loading existing index: {persist_dir!r}  "
              f"(pass force_rebuild=True to regenerate)")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    print(f"Building index '{collection_name}' "
          f"from {len(documents)} chunks  →  {persist_dir!r} ...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"Index saved: {persist_dir!r}")
    return db


def load_index(
    persist_dir: str,
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
) -> Chroma:
    """Load a previously persisted ChromaDB index."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


# ── Standalone demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from Chunking import (
        split_dataset,
        load_csv_corpus,
        create_chunks_strategy_a,
        create_chunks_strategy_b,
    )

    print("=" * 60)
    print("Part 1: Building Vector Database Indices")
    print("=" * 60)

    eval_df, knowledge_df = split_dataset()
    raw_docs = load_csv_corpus(knowledge_df)
    docs_a   = create_chunks_strategy_a(raw_docs)
    docs_b   = create_chunks_strategy_b(raw_docs)

    embeddings = get_embedding_model()

    print("\nBuilding Index A (fixed-size chunks) ...")
    db_a = build_index(docs_a, embeddings, CHROMA_DIR_A, COLLECTION_A)

    print("\nBuilding Index B (semantic chunks) ...")
    db_b = build_index(docs_b, embeddings, CHROMA_DIR_B, COLLECTION_B)

    # Quick sanity check
    test_q   = eval_df["prompt"].iloc[0]
    print(f"\nSanity check – query: {test_q[:70]}...")
    results = db_b.similarity_search(test_q, k=3)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:80]}...")

    print(f"\nIndex A: {CHROMA_DIR_A}")
    print(f"Index B: {CHROMA_DIR_B}")
    print("Both indices are ready.")
