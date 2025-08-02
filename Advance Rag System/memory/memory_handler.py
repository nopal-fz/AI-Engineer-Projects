from langchain.memory import ConversationBufferMemory
from utils.logger import setup_logger

logger = setup_logger(__name__)
# Function to create a memory handler for conversation
def create_memory_handler(memory_key="chat_history", return_messages=True):
    try:
        logger.info("Creating conversation buffer memory handler")
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages
        )
        logger.info("Successfully created conversation buffer memory handler")
        return memory
    except Exception as e:
        logger.error(f"Failed to create conversation buffer memory handler: {e}")
        return None