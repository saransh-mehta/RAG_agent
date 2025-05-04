DOC_PATH = "/Users/sarmehta/Desktop/RAG_agent/Docs/hdfc_policy_doc.pdf"
DOC_DIR = "/Users/sarmehta/Desktop/RAG_agent/Docs/"

#for text splitting 
CHUNK_SIZE = 300
OVERLAP = 30

# retrieval
EMBED_DIMS = 384#768
TOP_K = 2

#llm model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" #"thenlper/gte-base"
LLM_MODEL_NAME = "llama3-8b-8192"
MODEL_PROVIDER = "groq"

