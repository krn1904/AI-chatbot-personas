"""Streamlit web UI — a browser front-end over the same engine (engine.py).

Run with:  streamlit run app.py
"""

import streamlit as st
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

st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("Chat with your documents")

personas = list_personas()
default_index = personas.index(DEFAULT_PERSONA) if DEFAULT_PERSONA in personas else 0
persona = st.selectbox("Persona", personas, index=default_index)

# Per-persona memory: each persona keeps its own conversation, like browser tabs.
# Switching personas shows that persona's own thread instead of wiping everything.
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

    # Retrieve + build the augmented turn — the same engine the CLI uses.
    context, sources = get_relevant_context(persona, query)
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
