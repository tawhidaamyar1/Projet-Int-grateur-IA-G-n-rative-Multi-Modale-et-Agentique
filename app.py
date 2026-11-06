"""Interface Streamlit : upload PDF, indexation, questions/réponses."""
from pathlib import Path

import streamlit as st

from agent import ask, build_agent
from ingest import ingest

st.set_page_config(page_title="MaintiDoc – RAG multimodal", layout="wide")
st.title("MaintiDoc – RAG multimodal agentique")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
    st.session_state.history = []

with st.sidebar:
    up = st.file_uploader("Charger un PDF", type="pdf")
    if up and st.button("Indexer"):
        Path("data").mkdir(exist_ok=True)
        path = Path("data") / up.name
        path.write_bytes(up.getbuffer())
        with st.spinner("Parsing, CLIP, descriptions, PGVector..."):
            st.success(f"Indexé : {ingest(path)}")

for role, text, imgs in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        for i in imgs:
            st.image(i, width=400)

if q := st.chat_input("Posez votre question sur le document…"):
    st.session_state.history.append(("user", q, []))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit…"):
            answer, imgs = ask(st.session_state.agent, q)
        st.markdown(answer)
        for i in imgs:
            st.image(i, width=400)
    st.session_state.history.append(("assistant", answer, imgs))
