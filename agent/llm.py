"""LLM initialization + Langfuse tracing callback."""

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Silence the OTLP exporter error spam when Langfuse is not configured
logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("QCHEM_MODEL", "openrouter/anthropic/claude-sonnet-4.6"),
    base_url=os.getenv("QCHEM_BASE_URL", "https://inference.airi.net:46783/v1"),
    api_key=os.getenv("QCHEM_API_KEY", ""),
    temperature=0,
)


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler if credentials are configured, else None.

    Env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        from langfuse.callback import CallbackHandler
    return CallbackHandler()


__all__ = ["llm", "get_langfuse_handler"]
