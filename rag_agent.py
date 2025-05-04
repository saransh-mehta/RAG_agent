
import os
from parameters import (CHUNK_SIZE,
                        OVERLAP,
                        LLM_MODEL_NAME,
                         MODEL_PROVIDER,
                         EMBED_MODEL_NAME)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from langchain.chat_models import init_chat_model
from prompting import prompt_rag_agent
from parameters import TOP_K
from api_keys import GROQ_API_KEY

def build_rag_agent(documents):
    """
    Core component of the RAG agent.
    1. Creating a text splitter to split the documents into chunks.
    2. Creating an embedding model to convert text into vectors.
    3. Creating a vector store to store the embeddings.
    4. Creating an LLM model and prompt to generate responses.
    5. Creating a retriever to retrieve relevant documents based on the query.
    6. Creating a Lang chain to combine the LLM and the vector store.
    """
    # text Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                    chunk_overlap=OVERLAP)
    split_docs = text_splitter.split_documents(documents)

    #EMBEDDING MODEL
    embed_model = HuggingFaceEmbeddings(model_name = EMBED_MODEL_NAME)

    # FAISS Index as vector store and retriever
    vectorstore = FAISS.from_documents(split_docs, embed_model)
    retriever = vectorstore.as_retriever()

    ## llm model
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm_model = init_chat_model(LLM_MODEL_NAME,
                           model_provider=MODEL_PROVIDER)

    # RAG AGENT CHAIN   
    rag_agent_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_rag_agent
        | llm_model
        | StrOutputParser()
    )

    return rag_agent_chain, vectorstore


