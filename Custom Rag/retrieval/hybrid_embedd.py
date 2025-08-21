from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from retrieval.dense_embedd import create_ollama_embedding
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Function to create a hybrid retriever combining BM25 and Ollama embeddings
def create_hybrid_retriever(chunks):
    try:
        logger.info("Creating hybrid retriever")
        
        # Dense Retriever
        ollama_retriever = create_ollama_embedding(chunks)
        
        if ollama_retriever is None:
            logger.error("Failed to create Ollama retriever")
            return None
        
        # Sparse Retriever
        documents = [Document(page_content=chunk) for chunk in chunks]
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=word_tokenize
        )
        bm25_retriever.k = 5  # Set the number of top documents to retrieve
        
        # Combine both retrievers into a hybrid retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[ollama_retriever, bm25_retriever],
            weights=[0.5, 0.5]  # Equal weight for both retrievers
        )
        
        logger.info("Successfully created hybrid retriever")
        return hybrid_retriever
    except Exception as e:
        logger.error(f"Failed to create hybrid retriever: {e}")
        return None