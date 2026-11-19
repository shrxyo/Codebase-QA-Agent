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
        # Groq retired llama-3.1-8b-instant for free/dev tiers on 2026-08-16;
        # gpt-oss-20b is their recommended replacement.
        return "openai/gpt-oss-20b"
    return "llama3.1:8b"


def get_request_options() -> dict:
    """Provider-specific kwargs to pass to chat.completions.create().

    gpt-oss is a reasoning model: it writes hidden "thinking" tokens before the
    answer, and those count against max_tokens and delay the first visible
    token. Low effort is enough for summaries and code Q&A over retrieved
    context, and keeps latency down.
    """
    if os.getenv("LLM_PROVIDER", "ollama") == "groq":
        return {"reasoning_effort": "low"}
    return {}
