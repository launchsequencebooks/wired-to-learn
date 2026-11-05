# A chat interface for asking the book questions (run: streamlit run qa_app.py).
import streamlit as st
from qa import BookQA

st.set_page_config(page_title="Ask the book", layout="centered")
st.header("Ask *Wired to Learn*")
qa = st.cache_resource(BookQA)()                           # load the index and encoder once

if "history" not in st.session_state:
    st.session_state.history = []
for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)
if question := st.chat_input("Ask about neural networks..."):
    r = qa.answer(question)
    reply = r["answer"] + ("\n\n" + "  \n".join(f"*Source: {s}*" for s in r["sources"]) if r["sources"] else "")
    st.session_state.history += [("user", question), ("assistant", reply)]
    st.rerun()
