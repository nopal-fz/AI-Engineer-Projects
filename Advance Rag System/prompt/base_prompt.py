from langchain_core.prompts import ChatPromptTemplate
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Create the base prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
And make sure the anser is concise and relevant to the question.

Question:
{question}

Context:
{context}

Answer:
"""

# Function to create a base prompt template
def create_base_prompt():
    try:
        logger.info("Creating base prompt template")
        prompt = ChatPromptTemplate.from_template(template)
        logger.info("Successfully created base prompt template")
        return prompt
    except Exception as e:
        logger.error(f"Failed to create base prompt template: {e}")
        return None