from retrieval.hybrid_retriever import create_hybrid_retriever
from reranker.hf_reranker import rerank_documents
from utils.extract_pdf import extract_text_with_pypdf2, clean_text
from utils.splitter import split_text_into_chunks
from answer.answer_question import answer_question
from summarizer.summarizer import create_ollama_llm_with_summary

from utils.logger import setup_logger

logger = setup_logger(__name__)

pdf_path = "doc/bitcoin.pdf"
model_name = "llama3:latest"

try:
    # Step 1: Extract text from PDF
    content = extract_text_with_pypdf2(pdf_path)
    if not content:
        raise ValueError("No content extracted from the PDF file.")
    
    # Step 2: Clean and split the text into chunks
    cleaned_content = clean_text(content)
    chunks = split_text_into_chunks(cleaned_content)

    # Step 3: Create a hybrid retriever
    hybrid_retriever = create_hybrid_retriever(chunks)
    
    question = "What is Bitcoin?"

    # Step 4: Retrieve relevant documents
    retrieved_docs = hybrid_retriever.get_relevant_documents(question)

    if not retrieved_docs:
        raise ValueError("No documents retrieved for the question.")
    
    # Step 5: Rerank the retrieved documents
    reranked_docs = rerank_documents(question, retrieved_docs)

    if not reranked_docs:
        raise ValueError("No documents after reranking.")
    
    # Step 6: Answer the question using the reranked documents
    response = answer_question(question, reranked_docs, model_name)
    print(f"\Answer:\n{response}")

except Exception as e:
    logger.error(f"An error occurred: {e}")