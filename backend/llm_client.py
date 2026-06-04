import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "groq":
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info("LLM provider: groq (model: %s)", get_model_name())
    else:
        _client = OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
        logger.info("LLM provider: ollama (model: %s)", get_model_name())

    return _client


def get_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "groq":
        return "llama-3.1-8b-instant"
    return "llama3.1:8b"
