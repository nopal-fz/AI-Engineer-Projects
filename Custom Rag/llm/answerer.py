# llm/answerer.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def _system_text(lang: Literal["id","en"]="id") -> str:
    return (
        "Anda asisten yang hanya menjawab berdasarkan CONTEXT di bawah. "
        "Jika tidak ada di CONTEXT, katakan: \"Saya tidak menemukan jawabannya di konteks.\" "
        "Jawab ringkas dan sertakan [ctx#]. Jangan tampilkan chain-of-thought."
        if lang == "id" else
        "Answer strictly from CONTEXT. If missing, say: "
        "\"I couldn't find the answer in the provided context.\" "
        "Be concise and include [ctx#]. No chain-of-thought."
    )

def _join_context(docs):
    return "\n\n".join(f"[ctx{i+1}] {d.page_content}" for i, d in enumerate(docs, 1))

def build_end2end_rag_chain(chat_model, retriever, language: Literal["id","en"]="id"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_text(language)),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])

    # Input chain: `question` → ({docs, question}) → ({context, question}) → prompt → llm → str
    return (
        {
            "docs": retriever,                 # retriever akan dipanggil dg input `question`
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: {
            "context": _join_context(x["docs"]),
            "question": x["question"]
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )