from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

from flashrank import Ranker 
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Rebuild the model schema for Pydantic (fix for the error)
FlashrankRerank.model_rebuild()

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Function to create a reranker instance using FlashrankRerank
def create_reranked_retriever(base_retriever):
    try:
        logger.info("Creating reranker retriever using FlashrankRerank")
        reranker = FlashrankRerank(
            ranker=ranker,
            top_k=5
        )
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker
        )
        logger.info("Successfully created reranked retriever")
        return compression_retriever
    except Exception as e:
        logger.error(f"Failed to create reranked retriever: {e}")
        return None