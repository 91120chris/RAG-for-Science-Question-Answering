"""
HW3 Part 1: Indexing Pipeline - Data Loading & Chunking Strategies
Course: RNN and Transformer, Spring 2026

Data source: kaggle-llm-science-exam/train.csv (no external downloads)

Split logic:
  - Randomly sample 50 questions → eval_df    (for generation evaluation)
  - Remaining 150 questions      → knowledge_df (RAG knowledge corpus)

Chunking (token-based size via RecursiveCharacterTextSplitter):
  Method A: Fixed-size  (500 tokens, 10% overlap)
  Method B: Semantic/Recursive (1000 tokens, paragraph-first, 10% overlap)

Terminal setup:
  pip install langchain langchain-community langchain-huggingface
  pip install chromadb sentence-transformers torch pandas transformers
"""

import pandas as pd
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH   = "./kaggle-llm-science-exam/train.csv"
OPTION_COLS    = ["A", "B", "C", "D", "E"]
TOKENIZER_NAME = "BAAI/bge-m3"    # same model used for embeddings

EVAL_SEED = 42
N_EVAL    = 50

# Method A – fixed-size tokens
CHUNK_SIZE_A    = 500    # tokens
CHUNK_OVERLAP_A = 50     # tokens (10 %)

# Method B – semantic / recursive tokens
CHUNK_SIZE_B    = 1000   # tokens
CHUNK_OVERLAP_B = 100    # tokens (10 %)


# ── Tokenizer (shared, loaded once) ────────────────────────────────────────────

_tokenizer = None

def get_tokenizer() -> AutoTokenizer:
    """Load BAAI/bge-m3 tokenizer once and cache it."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def token_length(text: str) -> int:
    """Return the number of tokens in *text* (no special tokens)."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ── Data loading & splitting ────────────────────────────────────────────────────

def split_dataset(
    path: str   = DATASET_PATH,
    n_eval: int = N_EVAL,
    seed: int   = EVAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train.csv and split into evaluation set and knowledge corpus.

    Returns
    -------
    eval_df      : n_eval rows sampled at random (reproducible via seed)
    knowledge_df : remaining rows used as the RAG corpus
    """
    df           = pd.read_csv(path)
    eval_df      = df.sample(n=n_eval, random_state=seed).reset_index(drop=True)
    knowledge_df = df.drop(index=eval_df.index).reset_index(drop=True)

    print(f"Dataset split  (seed={seed}):")
    print(f"  Eval set  : {len(eval_df)} questions  (for generation evaluation)")
    print(f"  Knowledge : {len(knowledge_df)} questions  (RAG corpus)")
    return eval_df, knowledge_df


def load_csv_corpus(knowledge_df: pd.DataFrame) -> list[Document]:
    """
    Convert each row of knowledge_df into a LangChain Document.

    Document format (~ 218 tokens / 873 chars per row):
        Question: <prompt>
        Options:
        A. <option_a>
        B. <option_b>
        C. <option_c>
        D. <option_d>
        E. <option_e>

    Metadata: source, row_id, answer (correct letter, for reference only).
    """
    documents = []
    for _, row in knowledge_df.iterrows():
        options_text = "\n".join(f"{col}. {row[col]}" for col in OPTION_COLS)
        content = (
            f"Question: {row['prompt']}\n"
            f"Options:\n{options_text}"
        )
        documents.append(Document(
            page_content=content,
            metadata={
                "source": "train_csv",
                "row_id": int(row["id"]),
                "answer": row["answer"],
            },
        ))

    avg_chars  = sum(len(d.page_content) for d in documents) / max(len(documents), 1)
    avg_tokens = sum(token_length(d.page_content) for d in documents) / max(len(documents), 1)
    print(f"Corpus ready: {len(documents)} documents, "
          f"avg {avg_chars:.0f} chars / {avg_tokens:.0f} tokens each")
    return documents


# ── Chunking strategies ─────────────────────────────────────────────────────────

def create_chunks_strategy_a(documents: list[Document]) -> list[Document]:
    """
    Method A – Fixed-size chunking.
    chunk_size = 500 tokens, overlap = 50 tokens (10 %).
    Risk: may cut mid-sentence / mid-option, losing context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_A,
        chunk_overlap=CHUNK_OVERLAP_A,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Method A – Fixed-size    : {CHUNK_SIZE_A} tokens, overlap {CHUNK_OVERLAP_A} tokens "
          f"-> {len(chunks)} chunks")
    return chunks


def create_chunks_strategy_b(documents: list[Document]) -> list[Document]:
    """
    Method B – Semantic / Recursive chunking.
    Splits on paragraph breaks first, then sentences, then words.
    chunk_size = 1000 tokens, overlap = 100 tokens (10 %).
    Better preserves option boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_B,
        chunk_overlap=CHUNK_OVERLAP_B,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Method B – Semantic/Rec. : {CHUNK_SIZE_B} tokens, overlap {CHUNK_OVERLAP_B} tokens "
          f"-> {len(chunks)} chunks")
    return chunks


# ── Standalone demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: Chunking & Indexing Pipeline")
    print("=" * 60)

    eval_df, knowledge_df = split_dataset()

    # Sanity check: no overlap between sets
    eval_ids      = set(eval_df["id"])
    knowledge_ids = set(knowledge_df["id"])
    assert eval_ids & knowledge_ids == set(), "Overlap detected!"
    assert len(eval_df) + len(knowledge_df) == pd.read_csv(DATASET_PATH).shape[0]
    print("Sanity check passed.\n")

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
