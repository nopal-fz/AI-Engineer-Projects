import os
import io
import hashlib
import tiktoken
from typing import Literal, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from utils.splitter import split_text_with_chonkie
from utils.pdf_reader import extract_text_from_pdf, clean_text
from retrieval.vectorstore_factory import build_vectorstore
from retrieval.hybrid_embedd import create_hybrid_retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils.logger import setup_logger
logger = setup_logger(__name__)

app = FastAPI(title="Customizable RAG API", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables for storing index
current_retriever = None
current_index_signature = None

OLLAMA_MODELS = ["mistral:latest", "phi4-reasoning:latest", "llama3:latest", "deepseek-r1:7b"]
OPENAI_MODELS = ["gpt-4o-mini"]

class RAGConfig(BaseModel):
    chunker_type: str = "Sentence Chunker"
    unit: str = "token"
    chunk_size: int = 300
    overlap: int = 50
    chonkie_mode: str = "semantic"
    embed_name: str = "nomic-embed-text"
    backend: str = "faiss"
    top_k: int = 5
    provider: str = "ollama"
    llm_model: str = "mistral:latest"
    language: str = "id"

class QueryRequest(BaseModel):
    question: str
    provider: str
    llm_model: str
    language: str

def create_chat_model(provider: Literal["ollama","openai"], model_name: str, temperature=0.2):
    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature, num_ctx=4096)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
    return ChatOpenAI(model=model_name, temperature=temperature)

def _system_text(lang: Literal["id","en"]="id") -> str:
    return (
        "Anda asisten yang hanya menjawab berdasarkan CONTEXT di bawah. "
        "Jika jawabannya tidak ada di CONTEXT, katakan: "
        "\"Saya tidak menemukan jawabannya di konteks.\" "
        "Jawab ringkas dan sertakan referensi [ctx#] bila relevan. "
        "Jangan tampilkan langkah penalaran/chain-of-thought."
        if lang == "id" else
        "You must answer strictly from the CONTEXT below. "
        "If not in the CONTEXT, say: \"I couldn't find the answer in the provided context.\" "
        "Be concise and include [ctx#] when relevant. No chain-of-thought."
    )

def _join_context(docs):
    return "\n\n".join(f"[ctx{i+1}] {d.page_content}" for i, d in enumerate(docs, 1))

def build_end2end_rag_chain(chat_model, retriever, language: Literal["id","en"]="id"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_text(language)),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])
    return (
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: {"context": _join_context(x["docs"]), "question": x["question"]})
        | prompt
        | chat_model
        | StrOutputParser()
    )

def compute_signature(raw_bytes: bytes, cfg: dict) -> str:
    h = hashlib.sha256()
    h.update(raw_bytes)
    h.update(str(sorted(cfg.items())).encode("utf-8"))
    return h.hexdigest()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def get_models():
    return {
        "ollama": OLLAMA_MODELS,
        "openai": OPENAI_MODELS
    }

@app.post("/api/build-index")
async def build_index(
    file: UploadFile = File(...),
    chunker_type: str = Form("Sentence Chunker"),
    unit: str = Form("token"),
    chunk_size: int = Form(300),
    overlap: int = Form(50),
    chonkie_mode: str = Form("semantic"),
    embed_name: str = Form("nomic-embed-text"),
    backend: str = Form("faiss"),
    top_k: int = Form(5)
):
    global current_retriever, current_index_signature
    
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File harus berformat PDF")
        
        pdf_bytes = await file.read()
        raw_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        cleaned = clean_text(raw_text)
        
        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="Teks kosong (mungkin PDF hasil scan)")
        
        # Chunk text
        token_len_fn = None
        if unit == "token":
            enc = tiktoken.get_encoding("cl100k_base")
            token_len_fn = lambda s: len(enc.encode(s))
        
        chunks = split_text_with_chonkie(
            text=cleaned,
            chunker_type=chunker_type,
            chunk_size=chunk_size,
            overlap=overlap,
            unit=unit,
            chonkie_mode=chonkie_mode,
            token_len_fn=token_len_fn,
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Gagal membuat chunks")
        
        # Build vectorstore
        vs = build_vectorstore(
            chunks=chunks,
            embed_name=embed_name,
            backend=backend,
            persist_dir=None,
            collection_name="rag_docs",
        )
        
        current_retriever = vs.as_retriever(search_kwargs={"k": top_k})
        
        cfg_sig = {
            "chunker_type": chunker_type,
            "unit": unit,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chonkie_mode": chonkie_mode,
            "embed_name": embed_name,
            "backend": backend,
            "top_k": top_k,
        }
        current_index_signature = compute_signature(pdf_bytes, cfg_sig)
        
        return {
            "status": "success",
            "message": f"Index berhasil dibuat dengan {len(chunks)} chunks",
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_rag(request: QueryRequest):
    global current_retriever
    
    if current_retriever is None:
        raise HTTPException(status_code=400, detail="Index belum dibuat. Upload PDF dan build index terlebih dahulu.")
    
    try:
        llm = create_chat_model(
            provider=request.provider,
            model_name=request.llm_model,
            temperature=0.2
        )
        
        chain = build_end2end_rag_chain(
            llm, 
            current_retriever, 
            language=request.language
        )
        
        answer = chain.invoke(request.question)
        
        return {
            "status": "success",
            "answer": answer,
            "question": request.question
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses query: {str(e)}")

@app.get("/api/status")
async def get_status():
    return {
        "index_built": current_retriever is not None,
        "index_signature": current_index_signature
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)