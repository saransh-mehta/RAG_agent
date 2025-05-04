from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

#document reader
def read_pdf(file_path):
    pdf_loader = PyPDFLoader(file_path)
    # will load each page as a document,
    document_pages = pdf_loader.load()
    return document_pages

# Agent State Definition
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
