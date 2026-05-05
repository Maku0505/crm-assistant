"""
config.py — Single source of truth for all settings.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
CHROMA_DIR = Path("chroma_db")

# ── Ollama models ─────────────────────────────────────────────────────────────
LLM_MODEL   = "mistral"          # change to "llama3" if you prefer
EMBED_MODEL = "nomic-embed-text"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 5      # number of chunks to retrieve per query
CHUNK_SIZE    = 8000   # large enough to keep most files as a single chunk
CHUNK_OVERLAP = 0      # no overlap needed when files are whole

# ── Generation ────────────────────────────────────────────────────────────────
TEMPERATURE = 0.2  # low = more factual, high = more creative