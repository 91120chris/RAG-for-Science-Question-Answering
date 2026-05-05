## Running the Pipeline

### Full Pipeline (recommended for first run)
Builds both ChromaDB indices from scratch, then runs all four steps:

```bash
python main.py
```

---

### Skip Index Rebuild (subsequent runs)

Reuses the existing `chroma_db_fixed/` and `chroma_db_semantic/` directories:

```bash
python main.py --skip-build
```

---

### Indexing + Retrieval Only (no LLM)

Runs Steps 1–3 without loading Mistral-7B. Useful for testing retrieval quality quickly:

```bash
python main.py --no-gen
```
