import os, os.path
import shutil
import pathlib
import streamlit as st
from parameters import DOC_PATH
from utils import read_pdf
import logging

from rag_agent import build_rag_agent_graph


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


st.set_page_config(page_title='🔗 RAG AI Agent')
st.title("RAG AI Agent Demo")

# #clearing existing files in document directory
# if DOC_PATH.exists():
#     shutil.rmtree(DOC_PATH)
    
# #os.makedirs(DOC_PATH)
# DOC_PATH.mkdir(parents=True, exist_ok=True)

# upload function for user
# the uploaded file is not saved in the directory/disk, its in memory
uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
print("uploaded pdf :", uploaded_pdf)

if "doc_text" not in st.session_state:
    st.session_state["doc_text"] = ""


if uploaded_pdf is not None:
    if st.button("Upload"):
        logger.info("PDF file uploaded.")
        try:
            
            pdf_file = read_pdf(uploaded_pdf)
            print("read pdf :", pdf_file)
            st.session_state["doc_text"]  = pdf_file
            if not st.session_state["doc_text"]:
                logger.warning("PDF uploaded but no extractable text found.")
                st.warning("No text found in PDF. Please upload a valid document.")
            else:
                logger.info("PDF processed successfully.")

                # here based on the document, we will already create
                # 1. the vector store (text splitting, embedding, faiss index)
                # 2. the state graph (retrieval and response nodes)
                
                rag_agent = build_rag_agent_graph(st.session_state["doc_text"])

                st.success("PDF processed successfully. Enter your query below.")
        except Exception as e:
            logger.exception("PDF processing failed.")
            st.error(str(e))

    if st.session_state["doc_text"]:
            query = st.text_input("Enter your question about the document:")
            if st.button("Submit"):

                if query.strip() == "":
                    logger.warning("User submitted an empty query.")
                    st.warning("Please enter a question.")

                else:
                    logger.info(f"Received query: {query}")

                    try:
                        with st.spinner("Finding the answer..."):

                            #result = qa_pipeline(question=query, context=doc_text)
                            result = rag_agent({"question": query})
                            st.markdown(f"**Answer:** {result['answer']}")

                            logger.debug(f"Model response: {result}")

                    except Exception as e:
                        logger.exception("Error during question answering.")
                        st.error(f"Error during QA: {e}")

    


