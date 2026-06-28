"""LLM initialization + Langfuse v3 tracing helpers."""

import logging
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Silence OTLP exporter errors when Langfuse is not configured
logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

load_dotenv()


def _endpoint_reachable(base_url: str, api_key: str = "", timeout: float = 3.0) -> bool:
    """Quick liveness probe for an OpenAI-compatible endpoint via GET /models."""
    if not base_url:
        return False
    import httpx

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
        # Auth failures (expired/invalid key) mean the endpoint is unusable —
        # treat as unreachable so we fall back. Other non-5xx answers are OK.
        if resp.status_code in (401, 403):
            return False
        return resp.status_code < 500
    except Exception:
        return False


def _build_llm() -> ChatOpenAI:
    """Use the primary endpoint when reachable; otherwise fall back to local Ollama.

    Ollama exposes an OpenAI-compatible API (default http://localhost:11434/v1),
    so the only thing that changes on fallback is the base URL and model name.
    Set QCHEM_OLLAMA_MODEL (and optionally QCHEM_OLLAMA_BASE_URL) to enable it.
    """
    primary_model = os.getenv("QCHEM_MODEL", "openrouter/anthropic/claude-sonnet-4-6")
    primary_base_url = os.getenv("QCHEM_BASE_URL", "https://inference.airi.net:46783/v1")
    primary_api_key = os.getenv("QCHEM_API_KEY", "")

    if _endpoint_reachable(primary_base_url, primary_api_key):
        logger.info("Using primary LLM endpoint %s (model=%s)", primary_base_url, primary_model)
        return ChatOpenAI(
            model=primary_model,
            base_url=primary_base_url,
            api_key=primary_api_key,
            temperature=0,
        )

    ollama_model = os.getenv("QCHEM_OLLAMA_MODEL", "")
    ollama_base_url = os.getenv("QCHEM_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if ollama_model:
        logger.warning(
            "Primary LLM endpoint %s unreachable — falling back to Ollama %s (model=%s)",
            primary_base_url, ollama_base_url, ollama_model,
        )
        return ChatOpenAI(
            model=ollama_model,
            base_url=ollama_base_url,
            api_key="ollama",  # Ollama ignores the key but ChatOpenAI requires non-empty
            temperature=0,
        )

    logger.error(
        "Primary LLM endpoint %s unreachable and QCHEM_OLLAMA_MODEL not set — "
        "agent calls will fail.", primary_base_url,
    )
    return ChatOpenAI(
        model=primary_model,
        base_url=primary_base_url,
        api_key=primary_api_key,
        temperature=0,
    )


llm = _build_llm()


def _is_langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_handler(session_id: str | None = None):
    """Return a Langfuse CallbackHandler, or None if not configured.

    Supports Langfuse SDK v2 and v3. In v3, session_id is propagated via
    langfuse_trace() context manager rather than the handler directly.
    """
    if not _is_langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler  # v3+
        return CallbackHandler()
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # v2
            return CallbackHandler(session_id=session_id)
        except Exception:
            return None
    except Exception:
        return None


@contextmanager
def langfuse_trace(session_id: str | None = None):
    """Context manager that opens a Langfuse trace with session_id.

    Langfuse v3: uses start_as_current_observation + propagate_attributes so
    the CallbackHandler automatically attaches under the correct trace/session.
    Langfuse v2: no-op context (session_id is set on the handler directly).
    Always flushes on exit.
    """
    if not _is_langfuse_configured():
        yield
        return

    _client = None
    try:
        from langfuse import Langfuse, propagate_attributes  # v3
        _client = Langfuse()
    except (ImportError, AttributeError):
        # v2 installed — context not needed, session_id lives on the handler
        yield
        _flush_v2_or_singleton()
        return
    except Exception:
        yield
        return

    # v3 path: wrap execution in an observation with session propagation
    try:
        obs_kw = {"name": "qchem-agent"}
        if session_id:
            with _client.start_as_current_observation(**obs_kw):
                with propagate_attributes(session_id=session_id):
                    yield
        else:
            with _client.start_as_current_observation(**obs_kw):
                yield
    finally:
        _client.flush()


def flush_langfuse(handler=None) -> None:
    """Flush pending Langfuse events. Safe to call with None.

    Handles both v3 (singleton client) and v2 (handler.flush()).
    """
    _flush_v2_or_singleton()
    # v2: also flush via the handler instance
    if handler and hasattr(handler, "flush"):
        try:
            handler.flush()
        except Exception:
            pass


def _flush_v2_or_singleton() -> None:
    """Flush the Langfuse singleton client (works for v3; no-op for v2 new instance)."""
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass


__all__ = ["llm", "get_langfuse_handler", "langfuse_trace", "flush_langfuse"]
