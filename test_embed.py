# test_embed.py
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import SentenceTransformerEmbeddings
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

@st.cache_resource
def load_embed():
    return HuggingFaceEmbeddings(model_name="thenlper/gte-base")

st.write("Loading embedding model...")
embed_model = load_embed()
st.success("Model loaded successfully.")