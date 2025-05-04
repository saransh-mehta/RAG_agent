import os
import shutil
import streamlit as st
from utils import read_pdf
from rag_agent import build_rag_agent
from parameters import DOC_DIR

# App config
st.set_page_config(page_title="🔗 RAG AI Agent", layout="centered")
st.title("📄 RAG AI Agent Demo")

#clearing existing files in document directory
if os.path.exists(DOC_DIR):
    shutil.rmtree(DOC_DIR)

# Session state for persistence
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_agent_chain" not in st.session_state:
    st.session_state.rag_agent_chain = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None


# Upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


# If new file uploaded
if uploaded_file and uploaded_file.name != st.session_state.pdf_path:

    file_path = os.path.join(DOC_DIR, uploaded_file.name)
    os.makedirs(DOC_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded.")

    # Load documents
    documents = read_pdf(file_path)
    st.write(f"Loaded {len(documents)} pages.")

    rag_agent_chain, vectorstore = build_rag_agent(documents)
    st.session_state.vectorstore = vectorstore
    st.session_state.rag_agent_chain = rag_agent_chain
    st.session_state.pdf_path = uploaded_file.name
    st.success("✅ RAG graph ready. Ask your questions below!")


# QA Section
if st.session_state.rag_agent_chain:
    st.subheader("Ask a question about your PDF")
    query = st.text_input("Your question", key="user_question")

    if query:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_agent_chain.invoke(query)
            st.write("🧠 Answer:")
            st.success(answer)

elif not uploaded_file:
    st.info("Please upload a PDF to get started.")