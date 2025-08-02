from prompt.base_prompt import create_base_prompt
from prompt.summa_prompt import create_summary_prompt
from langchain_ollama.llms import OllamaLLM
from utils.logger import setup_logger
import re

def clean_llm_output(text):
    # Hapus tag HTML-like
    return re.sub(r'<[^<>]+>', '', text).strip()

logger = setup_logger(__name__)

# Function to answer a question using the hybrid retriever
def answer_question(question, documents, model_name="deepseek-r1:7b", memory=None):
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # If memory is provided, append past chat history to the context
        past_memory = ""
        if memory:
            past = memory.load_memory_variables({})
            past_memory = past.get("chat_history", "")
        
        context = past_memory + "\n\n" + context
        
        # QA step
        base_prompt = create_base_prompt()
        
        logger.info("Creating Ollama LLM instance for answering question")
        llm = OllamaLLM(model=model_name)
        
        qa_chain = base_prompt | llm
        base_answer = qa_chain.invoke({"question": question, "context": context})
        base_answer = clean_llm_output(base_answer)
        logger.info("Generated base answer")
        logger.debug(f"Base answer: {base_answer}")
        
        # Summarization Step
        summary_prompt = create_summary_prompt()
        summa_chain = summary_prompt | llm
        summarized_answer = summa_chain.invoke({"original_answer": base_answer})
        summarized_answer = clean_llm_output(summarized_answer)
        logger.info("Generated summarized answer")
        logger.debug(f"Summarized answer: {summarized_answer}")
        
        # If memory is provided, save the context and response
        if memory:
            memory.save_context({"input": question}, {"output": summarized_answer})
        
        logger.info("Successfully generated answer")
        return summarized_answer
    
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return None