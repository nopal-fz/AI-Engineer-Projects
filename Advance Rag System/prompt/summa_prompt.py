from langchain_core.prompts import ChatPromptTemplate
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Create the base prompt template
template = """
You are an expert technical explainer.

Given the following technical explanation or abstract, provide a structured and concise summary with:

- A short, insightful paragraph for each section
- No repetition of the original answer
- Focus on clarity. Avoid excessive jargon unless absolutely necessary.

Original Answer:
{original_answer}

Now provide the summarized and structured explanation:
"""

# Function to create a summary prompt template
def create_summary_prompt():
    try:
        logger.info("Creating summary prompt template")
        prompt = ChatPromptTemplate.from_template(template)
        logger.info("Successfully created summary prompt template")
        return prompt
    except Exception as e:
        logger.error(f"Failed to create summary prompt template: {e}")
        return None