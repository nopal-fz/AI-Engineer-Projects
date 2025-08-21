# utils/pdf_reader.py
import io
from typing import Union
from utils.logger import setup_logger

logger = setup_logger(__name__)

def _is_pathlike(x) -> bool:
    return isinstance(x, (str, bytes))

def _to_fileobj(pdf_input: Union[str, bytes, io.BytesIO]) -> io.BytesIO:
    """
    Normalisasi input menjadi file-like (BytesIO).
    - Jika path (str/bytes): baca ke memori
    - Jika sudah BytesIO: langsung pakai
    """
    if isinstance(pdf_input, io.BytesIO):
        pdf_input.seek(0)
        return pdf_input
    if isinstance(pdf_input, (bytes, bytearray)):
        return io.BytesIO(pdf_input)
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return io.BytesIO(f.read())
    raise TypeError("pdf_input harus berupa path str/bytes, bytes, atau io.BytesIO")

def extract_text_from_pdf(pdf_input: Union[str, bytes, io.BytesIO], use_pdfminer_fallback: bool = True) -> str:
    """
    Ekstrak teks dari PDF.
    - Menerima path string, bytes, atau BytesIO.
    - Coba PyPDF2 dulu; jika kosong & use_pdfminer_fallback=True → coba pdfminer.six.
    """
    # Normalisasi ke file-like
    bio = _to_fileobj(pdf_input)

    # Coba dengan PyPDF2/pypdf
    try:
        import PyPDF2  # pypdf kompatibel API yang sama
        logger.info("Starting PDF text extraction via PyPDF2")
        reader = PyPDF2.PdfReader(bio)

        # Jika terenkripsi, coba decrypt tanpa password (beberapa PDF hanya "flagged")
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # may or may not work
            except Exception:
                pass

        texts = []
        for page in reader.pages:
            # bisa mengembalikan None
            t = page.extract_text()
            if t:
                texts.append(t)
        text = "\n".join(texts).strip()
        if text:
            return text
    except Exception as e:
        logger.error(f"PyPDF2 failed: {e}")

    # Fallback ke pdfminer untuk kasus yang lebih susah (bukan OCR)
    if use_pdfminer_fallback:
        try:
            logger.info("Falling back to pdfminer.six")
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            # pdfminer bisa langsung dari path atau fileobj; kita kasih bytes
            bio.seek(0)
            text = pdfminer_extract_text(bio) or ""
            return text.strip()
        except Exception as e:
            logger.error(f"pdfminer fallback failed: {e}")

    return ""

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", "")
    text = " ".join(text.split())
    return text