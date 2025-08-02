from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Function to split text into chunks using RecursiveCharacterTextSplitter
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    try:
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Successfully split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Failed to split text: {e}")
        return []