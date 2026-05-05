# Architecture

## How it works

The system has two phases: **ingestion** (run once to process your data) and **querying** (runs on every user message).

---

## Phase 1 — Ingestion

Run once with `python ingest.py`.

```
your data/
├── crm_records/
├── tickets/
├── emails/
└── ...
        │
        ▼
1. FILE LOADER
   Detects file type and extracts text.
   Supported: .txt .md .csv .pdf .docx .xlsx .eml .zip
   Zip files are automatically extracted before processing.
        │
        ▼
2. CHUNKER
   Splits each document into overlapping chunks.
   Size: 512 tokens | Overlap: 64 tokens
   Every chunk keeps metadata: filename, source_type, relative_path
        │
        ▼
3. EMBEDDER
   Converts each chunk into a 768-dimensional vector.
   Model: nomic-embed-text running locally via Ollama.
        │
        ▼
4. CHROMADB
   Stores vectors + text + metadata on disk (chroma_db/).
   Persists between sessions — no need to re-ingest unless data changes.
```

---

## Phase 2 — Query (every user message)

```
User types a question in Streamlit
        │
        ▼
1. EMBED QUERY
   The question is converted to a vector using the same
   nomic-embed-text model used during ingestion.
        │
        ▼
2. RETRIEVE
   ChromaDB finds the top-k most similar chunks by vector distance.
   Optional filter: narrow search to a specific source_type
   (e.g. only tickets, only emails, only crm records).
        │
        ▼
3. BUILD CONTEXT
   Retrieved chunks are assembled into a numbered context block.
   Each chunk is tagged with its source type and filename.
        │
        ▼
4. PROMPT MISTRAL
   System prompt + context + user question are sent to Mistral via Ollama.
   System prompt instructs: answer only from the context provided,
   cite which document your answer comes from,
   and say so clearly if the context is insufficient.
        │
        ▼
5. RESPONSE
   Answer is displayed in the Streamlit chat UI.
   A collapsible Sources panel lists the exact files used.
```

---

## Design decisions

**Ollama for both LLM and embeddings**
Using Ollama to serve Mistral and nomic-embed-text means the entire stack
runs as a local HTTP server with no Python ML dependencies. No PyTorch
version conflicts, no compiler requirements, no outbound network calls.

**nomic-embed-text instead of MiniLM**
nomic-embed-text produces 768-dimensional embeddings vs 384 from
all-MiniLM-L6-v2. Better retrieval quality at no extra cost since Ollama
already manages the model lifecycle.

**Metadata on every chunk**
Storing source_type, filename, and relative_path on each chunk serves
two purposes: it enables filtered retrieval at query time (the user can
scope a search to just CRM records or just emails), and it powers the
sources panel so every answer is traceable to a specific document.

**Single config file**
All tuneable parameters — model names, top-k, chunk size, temperature —
live in config.py and are imported by both ingest.py and app.py.
One place to change anything before or during a demo.

**Grounded system prompt**
The LLM is explicitly told to answer only from the provided context.
If the retrieved chunks don't contain enough information, it says so
rather than guessing. This keeps answers honest and auditable.

---

## File map

```
config.py      — all settings (imported by ingest + app)
ingest.py      — phase 1: load → chunk → embed → store
app.py         — phase 2: query UI + retrieval + generation
chroma_db/     — persisted vector store (auto-generated, not in git)
data/          — your dataset (not in git)
```
