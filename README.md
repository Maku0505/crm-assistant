# CRM Assistant

A local RAG-powered assistant for querying internal CRM data using natural language.
Runs fully offline — no OpenAI or external APIs required.

## Stack

| Layer | Tool |
|---|---|
| LLM | Mistral 7B via Ollama |
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
git clone https://github.com/Maku0505/crm-assistant.git
cd crm-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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
ollama pull mistral
ollama pull nomic-embed-text
```

**4. Add your dataset**

Copy the dataset folders into the `data/` directory:

```
crm-assistant/
└── data/
    ├── crm_records/
    ├── sales/
    ├── tickets/
    ├── emails/
    ├── documents/
    └── files/
```

Supported file types: `.txt`, `.md`, `.csv`, `.pdf`, `.docx`, `.xlsx`, `.eml`, `.msg`, `.zip`

Zip files are automatically extracted on ingestion.

**5. Ingest the dataset**

```bash
python ingest.py
```

This chunks all documents, embeds them, and stores them in ChromaDB.
Run once — takes 5–20 minutes depending on dataset size.

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
| `LLM_MODEL` | `mistral` | Ollama model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Token size per chunk during ingestion |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TEMPERATURE` | `0.2` | LLM temperature (0 = factual, 1 = creative) |

## Example queries

- "Summarize the history of Acme Corp before my meeting"
- "What is the SLA for critical support tickets?"
- "Draft a reply to the latest complaint email"
- "Which leads are still open from last quarter?"
- "What channels does the platform support?"

## Project structure

```
crm-assistant/
├── ingest.py      # Load, chunk, embed, and store documents → ChromaDB
├── app.py         # Streamlit chat UI
├── config.py      # Centralised settings
└── data/          # Your dataset goes here
```
