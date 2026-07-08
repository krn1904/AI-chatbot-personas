"""Streamlit web UI — a browser front-end over the same engine (engine.py).

Run with:  streamlit run app.py
"""

import os

import streamlit as st

st.set_page_config(page_title="Chat with your documents", page_icon="📄")

# API key: Streamlit Cloud provides it via st.secrets; locally the engine reads .env.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:
    pass  # no secrets.toml locally — the engine falls back to .env

from google.genai import types

from engine import (
    DEFAULT_PERSONA,
    AnswerStream,
    build_config,
    get_relevant_context,
    list_personas,
    load_persona,
    with_context,
)
from ingest import build_memory_index, ensure_indexes


@st.cache_resource
def _startup() -> bool:
    """Rebuild baked-in indexes once per deploy — a fresh container has none."""
    ensure_indexes()
    return True


_startup()

st.title("Chat with your documents")

# --- Sidebar: upload your own documents (session-only, held in memory) ---
st.sidebar.header("Your documents")
uploaded = st.sidebar.file_uploader(
    "Upload files to chat with (this session only)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

# Re-index only when the set of uploaded files changes (embedding costs API calls).
signature = tuple((f.name, f.size) for f in uploaded) if uploaded else ()
if st.session_state.get("upload_sig") != signature:
    st.session_state.upload_sig = signature
    if uploaded:
        with st.sidebar, st.spinner("Indexing your documents…"):
            files = [(f.name, f.getvalue()) for f in uploaded]
            st.session_state.upload_index = build_memory_index(files)
    else:
        st.session_state.upload_index = None

upload_index = st.session_state.get("upload_index")
if upload_index is not None:
    st.sidebar.success(f"{len(signature)} document(s) ready — your files answer first.")

# --- Persona picker ---
personas = list_personas()
default_index = personas.index(DEFAULT_PERSONA) if DEFAULT_PERSONA in personas else 0
persona = st.selectbox("Persona", personas, index=default_index)

# Per-persona memory: each persona keeps its own conversation, like browser tabs.
if "chats" not in st.session_state:
    st.session_state.chats = {}
chat = st.session_state.chats.setdefault(persona, {"messages": [], "history": []})

if st.button("Clear this chat", disabled=not chat["history"]):
    st.session_state.chats[persona] = {"messages": [], "history": []}
    st.rerun()

# Replay this persona's conversation so far.
for role, text, sources in chat["history"]:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            st.caption("📄 Sources: " + ", ".join(sources))

query = st.chat_input("Type your question…")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    chat["history"].append(("user", query, None))
    chat["messages"].append(types.UserContent(query))

    # Uploads win, then baked-in docs — the same engine the CLI uses.
    context, sources = get_relevant_context(persona, query, upload_index)
    call_contents = chat["messages"][:-1] + [with_context(context, query)]
    config = build_config(load_persona(persona))

    with st.chat_message("assistant"):
        stream = AnswerStream(call_contents, config)
        try:
            answer = st.write_stream(stream)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            chat["messages"].pop()  # roll back the dangling user turn
            answer = None

        if answer is not None:
            shown_sources = sources if stream.used_docs else []
            if shown_sources:
                st.caption("📄 Sources: " + ", ".join(shown_sources))
            chat["history"].append(("assistant", answer, shown_sources))
            chat["messages"].append(types.ModelContent(answer))
