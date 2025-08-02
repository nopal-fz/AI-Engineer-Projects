import PyPDF2
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Function to extract text from a PDF file using PyPDF2
def extract_text_with_pypdf2(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            logger.info(f"Opening PDF file: {pdf_path}")
            text = ''
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ''
                logger.debug(f"Extracted text from page {i+1}, length: {len(page_text)}")
                text += page_text
            logger.info(f"Successfully extracted text from PDF: {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""
    
# Function clean up the extracted text
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    return text