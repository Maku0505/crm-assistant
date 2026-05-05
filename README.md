# CRM Assistant

A local RAG-powered assistant for querying internal CRM data using natural language.
Runs fully offline — no OpenAI or external APIs required.

## Stack

| Layer | Tool |
|---|---|
| LLM | Gemma2 2B via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB |
| Orchestration | LangChain |
| UI | Streamlit |

## Requirements

- Python 3.11
- [Ollama](https://ollama.com) installed and running

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <your-repo-url>
cd crm-assistant
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install langchain langchain-community langchain-core langchain-ollama \
            langchain-text-splitters chromadb pypdf \
            python-docx openpyxl streamlit \
            "sentence-transformers==2.7.0" "transformers==4.40.0"
```

**3. Pull Ollama models**

```bash
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

**4. Add your dataset**

Copy the dataset folders into the `data/` directory:

```
crm-assistant/
└── data/
    ├── crm_records/      → customers.csv, leads.csv
    ├── sales/            → sales_notes.csv, meeting notes
    ├── tickets/          → support_tickets.csv
    ├── emails/           → email_threads.json
    ├── documets/         → policy, FAQ, and service docs
    └── files/            → proposals and implementation docs
```

Supported file types: `.txt`, `.md`, `.csv`, `.pdf`, `.docx`, `.xlsx`, `.eml`, `.msg`, `.json`, `.zip`

Zip files are automatically extracted on ingestion.

**5. Ingest the dataset**

```bash
python ingest.py
```

Each file is stored as a complete document — no mid-file splitting — so every
query retrieves the full content of the relevant file.

To rebuild from scratch:

```bash
python ingest.py --reset
```

**6. Start the app**

Make sure Ollama is running in a separate terminal:

```bash
ollama serve
```

Then launch the UI:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma2:2b` | Ollama model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `TOP_K` | `5` | Number of documents retrieved per query |
| `CHUNK_SIZE` | `8000` | Max chars before a file is split |
| `TEMPERATURE` | `0.2` | LLM temperature (0 = factual, 1 = creative) |

## Features

- Natural language Q&A over all internal CRM data
- Source citations on every answer — see exactly which file was used
- Filter retrieval by source type (crm, ticket, email, sales, document)
- Adjustable top-k and temperature from the sidebar
- One-click example questions
- Fully offline — no data leaves your machine

## Example queries

- "What is the biggest company by number of employees?"
- "Summarize Vertex Telecom before my meeting"
- "What is the SLA for critical support tickets?"
- "Which leads have high urgency?"
- "Draft a reply to the latest complaint email"
- "What channels does the AllMessage platform support?"
- "Which customers are assigned to Omar?"

## Project structure

```
crm-assistant/
├── ingest.py        — load, embed, and store all documents into ChromaDB
├── app.py           — Streamlit chat UI + query pipeline
├── config.py        — all settings in one place
├── README.md        — this file
├── ARCHITECTURE.md  — design decisions and data flow
└── data/            — your dataset (not committed to git)
```

## Notes

- Run `python ingest.py --reset` any time the dataset changes
- The `chroma_db/` folder is auto-generated and excluded from git
- `gemma2:2b` is used for speed on CPU-only machines; swap to `gemma2:9b`
  in `config.py` for better quality if your machine can handle it