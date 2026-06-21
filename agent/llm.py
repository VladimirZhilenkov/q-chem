"""LLM initialization + Langfuse v3 tracing helpers."""

import logging
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Silence OTLP exporter errors when Langfuse is not configured
logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("QCHEM_MODEL", "openrouter/anthropic/claude-sonnet-4-6"),
    base_url=os.getenv("QCHEM_BASE_URL", "https://inference.airi.net:46783/v1"),
    api_key=os.getenv("QCHEM_API_KEY", ""),
    temperature=0,
)


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
