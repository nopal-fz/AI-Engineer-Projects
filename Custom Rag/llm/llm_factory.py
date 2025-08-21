# llm/llm_factory.py
from typing import Literal, Optional, Dict, Any
from utils.logger import setup_logger

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # butuh OPENAI_API_KEY

logger = setup_logger(__name__)

# Model yang kita expose ke user
OLLAMA_MODELS = {
    "mistral:latest",
    "phi4-reasoning:latest",
    "llama3:latest",
    "deepseek-r1:7b",
}
OPENAI_MODELS = {"gpt-4o-mini"}

Provider = Literal["ollama", "openai"]

def create_chat_model(
    provider: Provider = "ollama",
    model_name: str = "mistral:latest",
    temperature: float = 0.2,
    timeout: Optional[int] = None,
    **kwargs: Any,
):
    """
    Mengembalikan ChatModel LangChain (ChatOllama / ChatOpenAI) sesuai provider & model.
    - provider='ollama' → pastikan Ollama sudah jalan dan model sudah di-pull.
    - provider='openai' → butuh env OPENAI_API_KEY.
    """
    logger.info(f"Create chat model: provider={provider}, model={model_name}, temp={temperature}")

    if provider == "ollama":
        if model_name not in OLLAMA_MODELS:
            raise ValueError(f"Model '{model_name}' tidak ada di daftar OLLAMA_MODELS: {OLLAMA_MODELS}")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=kwargs.get("num_ctx", 4096),
            # kamu bisa tambahkan param khusus ollama lain di kwargs
            # e.g. mirostat, top_p, repeat_penalty, dll.
        )

    # provider == "openai"
    if model_name not in OPENAI_MODELS:
        raise ValueError(f"Model '{model_name}' tidak ada di daftar OPENAI_MODELS: {OPENAI_MODELS}")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        # streaming=kwargs.get("streaming", False),  # aktifkan kalau butuh
    )