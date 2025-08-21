from typing import List, Optional
from utils.logger import setup_logger
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = setup_logger(__name__)

EMBED_CHOICES = {"text-embedding-3-small", "nomic-embed-text"}

def create_dense_retriever(
    chunks: List[str],
    embed_name: str = "nomic-embed-text",   # "nomic-embed-text" | "text-embedding-3-small"
    top_k: int = 5
):
    """
    Bangun retriever dense dengan pilihan embedding:
    - "nomic-embed-text" (Ollama)
    - "text-embedding-3-small" (OpenAI)
    """
    try:
        if embed_name not in EMBED_CHOICES:
            raise ValueError(f"embed_name harus salah satu dari {EMBED_CHOICES}")
        if not chunks:
            raise ValueError("chunks kosong.")

        logger.info(f"Creating dense retriever (embed={embed_name}, top_k={top_k})")

        # Pilih embedding backend
        if embed_name == "text-embedding-3-small":
            embedding = OpenAIEmbeddings(model=embed_name)  # perlu OPENAI_API_KEY
        else:
            embedding = OllamaEmbeddings(model=embed_name)

        # Vector store in-memory
        vectorstore = InMemoryVectorStore(embedding=embedding)
        documents = [Document(page_content=c) for c in chunks]
        vectorstore.add_documents(documents)

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        logger.info("Successfully created dense retriever")
        return retriever

    except Exception as e:
        logger.exception(f"Failed to create dense retriever: {e}")
        return None

# Backward-compat (jika ada kode lama yang masih memanggil ini)
def create_ollama_embedding(chunks: List[str], top_k: int = 5):
    """Alias lama → default ke Ollama nomic-embed-text."""
    return create_dense_retriever(chunks, embed_name="nomic-embed-text", top_k=top_k)