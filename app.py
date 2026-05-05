"""
app.py — Streamlit CRM assistant UI.
Run with:  streamlit run app.py
"""

import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

import config

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CRM Assistant", page_icon="💼", layout="wide")
st.title("💼 CRM Assistant")
st.caption(f"Powered by {config.LLM_MODEL} · {config.EMBED_MODEL} · ChromaDB")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    source_filter = st.multiselect(
        "Filter by source type",
        options=["crm", "sales", "ticket", "email", "document", "file", "general"],
        default=[],
        help="Leave empty to search all sources.",
    )

    top_k = st.slider("Results to retrieve (top-k)", min_value=1, max_value=10,
                      value=config.TOP_K)

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0,
                            value=config.TEMPERATURE, step=0.05)

    st.divider()
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Example questions**")
    examples = [
        "Summarize the history of Acme Corp",
        "What is the SLA for critical support tickets?",
        "Draft a reply to the latest complaint email",
        "Which leads are still open from last quarter?",
        "What channels does the platform support?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex
            st.rerun()

# ── Load vector store (cached so it only loads once) ─────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_vectorstore():
    if not config.CHROMA_DIR.exists():
        return None
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embeddings,
    )

vectorstore = load_vectorstore()

if vectorstore is None:
    st.error("No knowledge base found. Run `python ingest.py` first, then restart the app.")
    st.stop()

# ── LLM (cached) ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_llm(model: str, temp: float):
    return ChatOllama(model=model, temperature=temp)

llm = load_llm(config.LLM_MODEL, temperature)

# ── Retrieval helper ──────────────────────────────────────────────────────────

def retrieve(query: str, k: int, source_types: list[str]):
    """Return (chunks, sources_markdown)."""
    if source_types:
        where = {"source_type": {"$in": source_types}}
        docs = vectorstore.similarity_search(query, k=k, filter=where)
    else:
        docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return [], "_No relevant documents found._"

    context_parts = []
    source_lines  = []
    seen = set()

    for i, doc in enumerate(docs, 1):
        meta  = doc.metadata
        fname = meta.get("filename", "unknown")
        stype = meta.get("source_type", "")
        rpath = meta.get("relative_path", fname)
        page  = meta.get("page", "")
        label = f"{rpath}" + (f" (p.{page})" if page else "")

        context_parts.append(f"[{i}] ({stype}) {doc.page_content.strip()}")

        if label not in seen:
            seen.add(label)
            source_lines.append(f"- `{label}`")

    sources_md = "\n".join(source_lines)
    return context_parts, sources_md

# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful CRM assistant for an internal sales and support team.
Answer questions using ONLY the context provided below.
If the context does not contain enough information, say so clearly — do not make things up.
When relevant, mention which document your answer comes from.
Be concise and professional."""

def build_messages(query: str, context_parts: list[str]):
    context_block = "\n\n".join(context_parts)
    user_content  = f"Context:\n{context_block}\n\nQuestion: {query}"
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

# ── Chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

# ── Handle input ──────────────────────────────────────────────────────────────

# Sidebar example buttons set this; chat_input also sets it
prompt = st.chat_input("Ask anything about your CRM data…")
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    with st.spinner("Searching knowledge base…"):
        context_parts, sources_md = retrieve(prompt, top_k, source_filter)

    # Generate
    with st.chat_message("assistant"):
        if not context_parts:
            answer = "I couldn't find any relevant information in the knowledge base for that question."
            st.markdown(answer)
        else:
            messages = build_messages(prompt, context_parts)
            with st.spinner("Thinking…"):
                response = llm.invoke(messages)
            answer = response.content
            st.markdown(answer)
            with st.expander("Sources"):
                st.markdown(sources_md)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources_md,
    })
