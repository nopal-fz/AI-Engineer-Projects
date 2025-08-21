from typing import List, Callable, Optional
import re

from utils.logger import setup_logger
from chonkie import SentenceChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)

MIN_SIZE, MAX_SIZE = 50, 2000
VALID_CHONKIE_MODES = {"semantic", "paragraph", "sentence"}
VALID_CHUNKER_TYPES = {"Sentence Chunker", "Recursive"}
VALID_UNITS = {"token", "character"}


def _validate_params(
    chunker_type: str,
    unit: str,
    chunk_size: int,
    overlap: int,
    chonkie_mode: str,
):
    if chunker_type not in VALID_CHUNKER_TYPES:
        raise ValueError(f"chunker_type harus salah satu dari {VALID_CHUNKER_TYPES}")

    if unit not in VALID_UNITS:
        raise ValueError(f"unit harus 'token' atau 'character'")

    if not (MIN_SIZE <= chunk_size <= MAX_SIZE):
        raise ValueError(f"chunk_size harus [{MIN_SIZE}..{MAX_SIZE}]")

    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap harus >= 0 dan < chunk_size")

    if chunker_type == "Sentence Chunker" and chonkie_mode not in VALID_CHONKIE_MODES:
        raise ValueError(f"chonkie_mode harus salah satu dari {VALID_CHONKIE_MODES}")


def _ensure_token_len_fn(token_len_fn: Optional[Callable[[str], int]]) -> Callable[[str], int]:
    if token_len_fn is not None:
        return token_len_fn
    # fallback: pakai tiktoken jika tersedia
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception as e:
        raise ValueError(
            "unit='token' dipilih, tetapi token_len_fn tidak diberikan dan tiktoken tidak tersedia."
        ) from e


def _pack_with_overlap(
    pieces: List[str],
    length_fn: Callable[[str], int],
    max_len: int,
    overlap: int,
    sep: str = "\n\n",
) -> List[str]:
    """
    Mengemas potongan (kalimat/paragraf) ke dalam chunks dengan batas panjang (token/char),
    menjaga overlap antar chunk.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def pieces_len(ps: List[str]) -> int:
        if not ps:
            return 0
        return length_fn(sep.join(ps))

    i = 0
    n = len(pieces)

    while i < n:
        candidate = cur + [pieces[i]]
        cand_len = pieces_len(candidate)
        if cand_len <= max_len or not cur:
            cur = candidate
            cur_len = cand_len
            i += 1
        else:
            # flush cur
            chunks.append(sep.join(cur))

            # siapkan overlap dari tail cur
            if overlap > 0:
                # ambil tail yang mendekati 'overlap'
                tail: List[str] = []
                j = len(cur) - 1
                while j >= 0 and pieces_len(tail + [cur[j]]) < overlap:
                    tail.insert(0, cur[j])
                    j -= 1
                cur = tail
                cur_len = pieces_len(cur)
            else:
                cur = []
                cur_len = 0

    if cur:
        chunks.append(sep.join(cur))

    # Jika overlap > 0, chunks bisa sedikit melebihi max_len karena penggabungan tail,
    # tapi pendekatan ini menjaga batas natural (paragraf/kalimat).
    return chunks


def split_text_with_chonkie(
    text: str,
    chunker_type: str = "Sentence Chunker",     # "Sentence Chunker" | "Recursive"
    chunk_size: int = 1000,
    overlap: int = 200,
    unit: str = "character",                    # "token" | "character"
    chonkie_mode: str = "semantic",             # "semantic" | "paragraph" | "sentence" (hanya utk Sentence Chunker)
    token_len_fn: Optional[Callable[[str], int]] = None,  # penghitung token jika unit="token"
) -> List[str]:
    """
    Memecah teks menjadi chunks berdasarkan opsi:
    1) Chunker Type:
       - "Sentence Chunker" (Chonkie): semantic-aware (mode: semantic|paragraph|sentence)
       - "Recursive" (LangChain): rule-based fixed length
    2) Unit Panjang:
       - "token" atau "character"
    3) Chunk Size & Overlap:
       - Validasi: 50 ≤ chunk_size ≤ 2000, 0 ≤ overlap < chunk_size
    """
    try:
        if not text:
            logger.warning("Input text kosong.")
            return []

        _validate_params(chunker_type, unit, chunk_size, overlap, chonkie_mode)
        logger.info(
            f"Splitting text (type={chunker_type}, unit={unit}, size={chunk_size}, "
            f"overlap={overlap}, mode={chonkie_mode})"
        )

        # Tentukan length function
        if unit == "token":
            length_fn = _ensure_token_len_fn(token_len_fn)
        else:
            length_fn = len

        if chunker_type == "Sentence Chunker":
            if chonkie_mode in {"semantic", "sentence"}:
                # Gunakan SentenceChunker bawaan chonkie
                counter = length_fn if unit == "token" else "character"
                chunker = SentenceChunker(
                    tokenizer_or_token_counter=counter,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
                pieces = chunker.chunk(text)
                chunks = [getattr(p, "text", str(p)) for p in pieces]
                logger.info(
                    f"Successfully split text with Chonkie SentenceChunker ({chonkie_mode}) into {len(chunks)} chunks"
                )
                return chunks

            elif chonkie_mode == "paragraph":
                # Mode paragraf: jaga batas paragraf lalu packing berdasarkan max_len & overlap
                # Split paragraf by 1+ blank lines
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
                if not paragraphs:
                    # fallback: treat whole text as one paragraph
                    paragraphs = [text.strip()]
                chunks = _pack_with_overlap(
                    paragraphs, length_fn, max_len=chunk_size, overlap=overlap, sep="\n\n"
                )
                logger.info(f"Successfully split text (Chonkie-paragraph mode) into {len(chunks)} chunks")
                return chunks

        else:  # "Recursive" (LangChain)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=length_fn,  # token-aware jika unit="token"
                add_start_index=True,
            )
            chunks = splitter.split_text(text)
            logger.info(
                f"Successfully split text with RecursiveCharacterTextSplitter into {len(chunks)} chunks"
            )
            return chunks

    except Exception as e:
        logger.exception(f"Failed to split text: {e}")
        return []