import os
import io
import hashlib
import textwrap
import streamlit as st
import PyPDF2
import tiktoken

from utils.splitter import split_text_with_chonkie
from utils.pdf_reader import extract_text_from_pdf, clean_text
from retrieval.vectorstore_factory import build_vectorstore
from retrieval.hybrid_embedd import create_hybrid_retriever

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils.logger import setup_logger
logger = setup_logger(__name__)

OLLAMA_MODELS = ["mistral:latest", "phi4-reasoning:latest", "llama3:latest", "deepseek-r1:7b"]
OPENAI_MODELS = ["gpt-4o-mini"]

def create_chat_model(provider: Literal["ollama","openai"], model_name: str, temperature=0.2):
    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature, num_ctx=4096)
    # openai
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

# UI
st.set_page_config(page_title="Customizable RAG – Streamlit", layout="wide")
st.title("🔎 Customizable RAG (Chonkie + FAISS/Chroma)")

with st.sidebar:
    st.header("📄 Data Source")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    st.header("✂️ Chunking")
    chunker_type = st.selectbox("Chunker Type", ["Sentence Chunker", "Recursive"])
    unit = st.radio("Unit Panjang", ["token", "character"], index=0)
    # default rekomendasi: 300 token / 1000 karakter
    default_chunk = 300 if unit == "token" else 1000
    default_overlap = 50 if unit == "token" else 200
    chunk_size = st.slider("Chunk Size (maxlen)", 50, 2000, default_chunk, step=10)
    overlap = st.slider("Overlap", 0, min(1999, chunk_size-1), default_overlap, step=5)
    chonkie_mode = st.selectbox("Chonkie Mode (khusus Sentence Chunker)",
                                ["semantic", "paragraph", "sentence"], index=0,
                                disabled=(chunker_type != "Sentence Chunker"))

    st.header("🧠 Embedding & Vector Store")
    embed_name = st.radio("Embedding", ["nomic-embed-text", "text-embedding-3-small"], index=0)
    backend = st.radio("Vector Store", ["faiss", "chroma"], index=0)
    top_k = st.slider("Top-K Retrieval", 1, 20, 5)

    st.header("🤖 LLM")
    provider = st.radio("Provider", ["ollama", "openai"], index=0)
    model_choices = OLLAMA_MODELS if provider == "ollama" else OPENAI_MODELS
    llm_model = st.selectbox("Model", model_choices, index=0)
    lang_choice = st.radio("Bahasa Jawaban", ["id", "en"], index=0)

    st.header("⚙️ Build / Rebuild Index")
    build_pressed = st.button("🔧 Build Index")

# Session State
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "index_signature" not in st.session_state:
    st.session_state.index_signature = None

# Build Index when requested
def compute_signature(raw_bytes: bytes, cfg: dict) -> str:
    h = hashlib.sha256()
    h.update(raw_bytes)
    h.update(str(sorted(cfg.items())).encode("utf-8"))
    return h.hexdigest()

if build_pressed:
    if not pdf_file:
        st.error("Mohon upload PDF terlebih dahulu.")
    else:
        pdf_bytes = pdf_file.getvalue()
        raw_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        cleaned = clean_text(raw_text)

        if not cleaned.strip():
            st.error("Teks kosong (mungkin PDF hasil scan). Pertimbangkan OCR.")
        else:
            # chunk
            try:
                # token length fn otomatis bila unit=token (pakai tiktoken)
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
            except Exception as e:
                st.exception(e)
                chunks = []

            if not chunks:
                st.error("Gagal membuat chunks.")
            else:
                # build dense retriever (non-hybrid). Jika ingin hybrid, aktifkan di sini.
                try:
                    vs = build_vectorstore(
                        chunks=chunks,
                        embed_name=embed_name,
                        backend=backend,
                        persist_dir=None,         # in-memory untuk Streamlit
                        collection_name="rag_docs",
                    )
                    retriever = vs.as_retriever(search_kwargs={"k": top_k})
                except Exception as e:
                    st.exception(e)
                    retriever = None

                if retriever:
                    st.session_state.retriever = retriever
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
                    st.session_state.index_signature = compute_signature(pdf_bytes, cfg_sig)
                    st.success(f"Index berhasil dibuat. Total chunks: {len(chunks)} ✅")

# Render Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
prompt = st.chat_input("Tanyakan sesuatu tentang dokumen…")
if prompt:
    if st.session_state.retriever is None:
        st.warning("Index belum dibuat. Upload PDF dan klik 'Build Index' dulu.")
    else:
        # tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # siapkan LLM & chain
        try:
            llm = create_chat_model(provider=provider, model_name=llm_model, temperature=0.2)
            chain = build_end2end_rag_chain(llm, st.session_state.retriever, language=lang_choice)
            answer = chain.invoke(prompt)
        except Exception as e:
            answer = f"Gagal memanggil LLM: {e}"

        # tampilkan jawaban
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)