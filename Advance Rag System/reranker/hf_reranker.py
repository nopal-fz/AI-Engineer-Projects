from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# intialize the CrossEncoder model
reranker = CrossEncoder(model_name)

# Function to rerank documents based on a question using CrossEncoder
def rerank_documents(question, documents):
    try:
        logger.info("Reranking documents with CrossEncoder")
        texts = [doc.page_content for doc in documents]
        pairs = [(question, text) for text in texts]
        scores = reranker.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs]
        logger.info(f"Successfully reranked {len(documents)} documents")
        return reranked_docs
    except Exception as e:
        logger.error(f"Failed to rerank documents: {e}")
        return documents  # fallback: return original order
