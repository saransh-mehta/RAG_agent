
import os
from parameters import (DOC_PATH, CHUNK_SIZE,
                        OVERLAP, EMBED_DIMS,
                        LLM_MODEL_NAME,
                         MODEL_PROVIDER,
                         EMBED_MODEL_NAME)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from api_keys import GROQ_API_KEY

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.storage import InMemoryStore

from langchain.chat_models import init_chat_model
from parameters import TOP_K
from utils import AgentState
from prompting import prompt
from langgraph.graph import StateGraph, START, END
import streamlit as st


def create_vector_store(document_pages):
    """
    Create a vector store from the document pages.
    """
    #TEXT SPLITTING
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE,
                                                chunk_overlap = OVERLAP)
    print("text splitter created")
    @st.cache_resource
    def init_embedding_model(embed_model_name):
        return HuggingFaceEmbeddings(model_name = embed_model_name)
    #embedding model
    #embed_model = HuggingFaceEmbeddings(model_name = EMBED_MODEL_NAME)
    embed_model = init_embedding_model(EMBED_MODEL_NAME)

    #embed_model = HuggingFaceInferenceAPIEmbeddings(
    #api_key=HUGGINGFACE_KEY,
    #model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("embedding model created")
    #VECTOR STORE
    faiss_index = faiss.IndexFlatL2(EMBED_DIMS)
    vector_store = FAISS(embedding_function = embed_model,
                        index = faiss_index,
                        docstore = InMemoryDocstore(),
                        index_to_docstore_id = {})
    print("faiss index created")
    # Split the documents into chunks
    #print("888888DOCUEMNT PAGES99999", document_pages[0])
    all_splits = text_splitter.split_documents(document_pages)
    print("documents split into chunks")
    # Add the chunks to the vector store
    document_ids = vector_store.add_documents(documents=all_splits)
    print("documents added to vector store")
    
    return vector_store

# node 1 for retriving context related to the question
def retrieval_node(state: AgentState):
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}

#node 2 for generating the answer
def response_node(state: AgentState):
    full_context = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": full_context})
    #print(messages)
    response = llm_model.invoke(messages)
    return {"answer": response.content}

def build_rag_agent_graph(document_pages):
    """
    Build the state graph for the RAG agent.
    """
    #LLM chat model
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm_model = init_chat_model(LLM_MODEL_NAME, model_provider= MODEL_PROVIDER)
    print("llm_model loaded")

    # creating the vector store
    vector_store = create_vector_store(document_pages)
    print("vector store created")
    # Create a state graph
    graph_builder = StateGraph(AgentState)
    print("graph builder created")
    # connecting nodes to build graph
    #START
    graph_builder.add_edge(START, "retrieval_node")
    print("edge added")
    # RETRIEVE NODE
    graph_builder.add_node("retrieval_node", retrieval_node)
    print("retrieval node added")
    #edge from retrival node to response node
    graph_builder.add_edge("retrieval_node", "response_node")
    print("edge added")
    # RESPONSE NODE
    graph_builder.add_node("response_node", response_node)
    print("response node added")
    #END
    graph_builder.add_edge("response_node", END)
    print("edge added")
    #compiling the graph
    graph = graph_builder.compile()
    print("graph compiled")
    return graph
