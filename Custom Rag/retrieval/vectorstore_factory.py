from typing import List, Literal, Optional
from utils.logger import setup_logger

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = setup_logger(__name__)

EMBED_CHOICES = {"text-embedding-3-small", "nomic-embed-text"}
BACKENDS = {"faiss", "chroma"}

def create_embedding(embed_name: str):
    if embed_name not in EMBED_CHOICES:
        raise ValueError(f"embed_name harus salah satu dari {EMBED_CHOICES}")
    if embed_name == "text-embedding-3-small":
        return OpenAIEmbeddings(model=embed_name)   # butuh OPENAI_API_KEY
    return OllamaEmbeddings(model=embed_name)

def build_vectorstore(
    chunks: List[str],
    embed_name: str = "nomic-embed-text",
    backend: Literal["faiss", "chroma"] = "faiss",
    persist_dir: Optional[str] = None,
    collection_name: str = "rag_docs",
):
    """
    Buat vectorstore FAISS/Chroma dari chunks.
    - FAISS: persist_dir = folder index (contoh: 'storage/faiss')
    - Chroma: persist_dir = folder DB (contoh: 'storage/chroma')
    """
    if not chunks:
        raise ValueError("chunks kosong.")
    if backend not in BACKENDS:
        raise ValueError(f"backend harus salah satu dari {BACKENDS}")

    embedding = create_embedding(embed_name)
    docs = [Document(page_content=c) for c in chunks]

    logger.info(f"Build vectorstore backend={backend}, embed={embed_name}, persist={persist_dir}")

    if backend == "faiss":
        vs = FAISS.from_documents(docs, embedding)
        # simpan hanya kalau persist_dir diberikan
        if persist_dir:
            vs.save_local(persist_dir)
        return vs

    # backend == "chroma"
    # pip install chromadb
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_dir,         # wajib kalau mau persist
        collection_name=collection_name,
    )
    # Chroma auto-persist saat instance dibuat
    return vs

def load_vectorstore(
    embed_name: str,
    backend: Literal["faiss", "chroma"] = "faiss",
    persist_dir: Optional[str] = None,
    collection_name: str = "rag_docs",
):
    """
    Load vectorstore dari disk.
    - FAISS: FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    - Chroma: Chroma(..., persist_directory=..., collection_name=...)
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend harus salah satu dari {BACKENDS}")
    if not persist_dir:
        raise ValueError("persist_dir harus diisi untuk load_vectorstore.")

    embedding = create_embedding(embed_name)

    logger.info(f"Load vectorstore backend={backend}, embed={embed_name}, persist={persist_dir}")

    if backend == "faiss":
        return FAISS.load_local(
            persist_dir,
            embedding,
            allow_dangerous_deserialization=True
        )

    # backend == "chroma"
    return Chroma(
        embedding_function=embedding,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )