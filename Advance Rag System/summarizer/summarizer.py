from langchain_ollama.llms import OllamaLLM
from prompt.summa_prompt import create_summary_prompt
from utils.logger import setup_logger

logger = setup_logger(__name__)

model_name = "deepseek-r1:7b"
# Function to create an Ollama LLM instance with a summary prompt
def create_ollama_llm_with_summary(original_answer: str):
    try:
        logger.info("Creating Ollama LLM instance with summary prompt")
        summary_prompt = create_summary_prompt()
        llm = OllamaLLM(model=model_name)
        
        chain = summary_prompt | llm
        logger.info("Successfully created Ollama LLM instance with summary prompt")
        return chain.invoke({"original_answer": original_answer})
    except Exception as e:
        logger.error(f"Failed to create Ollama LLM instance with summary prompt: {e}")
        return None