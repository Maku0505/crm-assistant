# Architecture

## Overview

A Retrieval-Augmented Generation (RAG) pipeline running fully locally.
Two phases: ingestion (runs once) and querying (runs on every user message).
No external APIs — everything runs on device via Ollama.

---

## Phase 1 — Ingestion

Run once with `python ingest.py`.

```
data/
├── crm_records/      customers.csv, leads.csv
├── sales/            sales_notes.csv, meeting_note_XX.txt
├── tickets/          support_tickets_80.csv
├── emails/           email_threads.json
├── documets/         policy docs, FAQ docs, service docs
└── files/            proposals, implementation docs
        │
        ▼
1. ZIP EXTRACTOR
   Automatically extracts any .zip files found in the dataset.
   Handles nested zips. Skips already-extracted folders.
        │
        ▼
2. FILE LOADER
   Detects file type and extracts clean text.
   .txt / .md   → read as plain text
   .csv         → loaded as one complete document (all rows preserved)
   .pdf         → extracted page by page
   .docx        → paragraphs joined into one document
   .xlsx        → each sheet becomes one document
   .eml / .msg  → headers + body extracted
   .json        → email threads parsed into individual documents
        │
        ▼
3. METADATA TAGGING
   Every document is tagged with:
   - source_type  (crm, sales, ticket, email, document, file)
   - filename     (customers.csv, support_sla.txt, ...)
   - relative_path (crm_records/customers.csv, ...)
   This enables filtered retrieval at query time.
        │
        ▼
4. STORAGE STRATEGY
   Files smaller than 8000 chars → stored as one complete document.
   Files larger than 8000 chars → split at paragraph boundaries.
   Goal: keep each file whole so queries return complete information.
        │
        ▼
5. EMBEDDER
   Each document is converted to a 768-dimensional vector.
   Model: nomic-embed-text running locally via Ollama.
   The vector captures the semantic meaning of the document.
        │
        ▼
6. CHROMADB
   Stores vectors + text + metadata on disk in chroma_db/.
   Persists between sessions — no need to re-ingest unless data changes.
```

---

## Phase 2 — Query

Runs every time the user sends a message in the Streamlit UI.

```
User types a question
        │
        ▼
1. EMBED QUERY
   The question is converted to a 768-dimensional vector
   using the same nomic-embed-text model used during ingestion.
   This puts the query in the same vector space as the documents.
        │
        ▼
2. SIMILARITY SEARCH (Document Retrieval)
   ChromaDB computes cosine similarity between the query vector
   and every stored document vector.
   Returns the top-k most semantically similar documents.

   Optional source_type filter — user can restrict search to:
   → only crm records, only tickets, only emails, etc.
        │
        ▼
3. BUILD CONTEXT
   Retrieved documents are assembled into a numbered context block.
   Each document is labelled with its source type and filename.
   Example:
     [1] (crm) customer_id,company_name,...
         CUST-038,Vertex Telecom,...
     [2] (ticket) TKT-002, Vertex Telecom, Email delivery...
        │
        ▼
4. PROMPT CONSTRUCTION
   Three parts are combined into one prompt sent to the LLM:
   - System prompt: defines the assistant role, data schema,
     field disambiguation, and grounding rules
   - Context block: the retrieved documents from step 3
   - User question: the original query
        │
        ▼
5. LLM GENERATION
   Gemma2 2B (via Ollama) generates a response based only
   on the provided context. It runs fully locally — no data
   leaves the machine.
        │
        ▼
6. RESPONSE + SOURCES
   Answer is displayed in the Streamlit chat UI.
   A collapsible Sources panel lists the exact files used,
   making every answer traceable to a specific document.
```

---

## Design decisions

**Ollama for both LLM and embeddings**
The entire stack runs as a local HTTP server with no Python ML
dependencies. No PyTorch version conflicts, no compiler requirements,
no outbound network calls. Ollama manages model loading and keeps
models warm in memory between requests.

**nomic-embed-text for embeddings**
Produces 768-dimensional embeddings vs 384 from all-MiniLM-L6-v2.
Better semantic representation at no extra cost since Ollama already
manages the model.

**Whole-file storage instead of fixed chunking**
Traditional RAG splits documents into fixed 512-token chunks. This
causes problems with structured data — a CSV with 50 customers split
into 10-row chunks means most queries only see a fraction of the data.
Instead, each file is stored as one complete document so every retrieval
returns the full file content.

**Metadata on every document**
Tagging each document with source_type, filename, and relative_path
serves two purposes: it enables filtered retrieval (search only tickets,
only CRM records) and powers the Sources panel so every answer is
traceable to a specific file.

**Schema-aware system prompt**
The system prompt contains the exact column names and field definitions
for every file in the dataset. This prevents the LLM from confusing
similar-sounding fields (e.g. company_size vs budget_range) and maps
common questions to the correct source files.

**Single config file**
All tuneable parameters — model names, top-k, chunk size, temperature —
live in config.py and are imported by both ingest.py and app.py.
One place to change anything before or during the demo.

**Gemma2 2B on Intel CPU**
Mistral 7B and larger models are too slow on a CPU-only Intel machine
(45-60 seconds per response). Gemma2 2B delivers acceptable speed
(10-15 seconds) while maintaining sufficient quality for CRM Q&A tasks.

---

## Data flow summary

```
data/ files
    └─► ZIP extractor ─► File loaders ─► Metadata tagger
                                               │
                                         nomic-embed-text
                                               │
                                          ChromaDB (disk)
                                               │
                              ┌────────────────┘
                              │
User query ─► nomic-embed-text ─► similarity search ─► top-k docs
                                                            │
                                                     system prompt
                                                            │
                                                      Gemma2 2B
                                                            │
                                                   answer + sources
```

---

## File map

```
config.py        all settings — imported by ingest.py and app.py
ingest.py        phase 1: load → tag → embed → store
app.py           phase 2: retrieve → prompt → generate → display
chroma_db/       persisted vector store (auto-generated, not in git)
data/            dataset (not in git)
```