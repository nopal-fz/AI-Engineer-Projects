import streamlit as st

from retrieval.hybrid_retriever import create_hybrid_retriever
from reranker.hf_reranker import rerank_documents
from utils.extract_pdf import extract_text_with_pypdf2, clean_text
from utils.splitter import split_text_into_chunks
from answer.answer_question import answer_question
#from summarizer.summarizer import create_ollama_llm_with_summary
from utils.logger import setup_logger

import os

# Setup logger
logger = setup_logger(__name__)

# Constants
pdfs_directory = "pdfs/"

deepseek = "deepseek-r1:7b"
llama = "llama3:latest"

model_name = llama  # Change to llama if needed

# Ensure 'pdfs/' directory exists
os.makedirs(pdfs_directory, exist_ok=True)

# Streamlit UI title
st.title("Advance RAG System")

# PDF Upload
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf",
    key="pdf_uploader",
    accept_multiple_files=False
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file:
    try:
        # Save file to folder pdfs/
        file_path = os.path.join(pdfs_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Uploaded file: {uploaded_file.name}")

        # Ekstraksi dan preprocessing
        content = extract_text_with_pypdf2(file_path)
        if not content.strip():
            st.error("No content extracted from the PDF file.")
            raise ValueError("Empty content from PDF.")
        logger.info("Text successfully extracted from PDF.")

        cleaned_content = clean_text(content)
        chunks = split_text_into_chunks(cleaned_content)
        logger.info("Text cleaned and split into chunks.")

        # Retriever
        hybrid_retriever = create_hybrid_retriever(chunks)
        logger.info("Hybrid retriever created.")
        
        # Chat input
        question = st.chat_input("Ask a question about the content of the PDF")
        
        if question:
            logger.info(f"Received question: {question}")
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})

            retrieved_docs = hybrid_retriever.get_relevant_documents(question)
            if not retrieved_docs:
                st.error("No documents retrieved.")
                raise ValueError("Retriever returned empty result.")
            logger.info(f"{len(retrieved_docs)} documents retrieved.")

            reranked_docs = rerank_documents(question, retrieved_docs)
            if not reranked_docs:
                st.error("No documents after reranking.")
                raise ValueError("Reranker returned empty result.")
            logger.info(f"{len(reranked_docs)} documents reranked.")

            response = answer_question(question, reranked_docs, model_name)
            if not response:
                st.error("Failed to generate answer.")
                raise ValueError("Model response is empty.")
            logger.info("Answer generated.")

            # summary = create_ollama_llm_with_summary(response)
            # logger.info("Summary created.")
            
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error("An error occurred while processing the PDF.")