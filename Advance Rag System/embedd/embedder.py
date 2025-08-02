from utils.logger import setup_logger
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

name_embed = "nomic-embed-text"

logger = setup_logger(__name__)

# Function to create an Ollama embedding instance
def create_ollama_embedding(chunks):
    try:
        logger.info("Creating Ollama embedding instance")
        embedding = OllamaEmbeddings(model=name_embed)
        vectorstore = InMemoryVectorStore(embedding=embedding)
        
        documents = [Document(page_content=chunk) for chunk in chunks]
        vectorstore.add_documents(documents)
        logger.info("Successfully created Ollama embedding instance")
        return vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Failed to create Ollama embedding instance: {e}")
        return None