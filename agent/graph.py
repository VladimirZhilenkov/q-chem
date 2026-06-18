"""ReAct agent with simple dialogue memory and logging.

Minimal architecture:
  1. PROMPT     — tells LLM how to think and call tools
  2. executor   — ReAct loop: Thought → Action → Observation
  3. _memory    — conversation history by session_id
  4. run_agent  — main entry point for UI and REPL
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from loguru import logger
import sys

from agent.llm import get_langfuse_handler, llm
from agent.registry import get_tools


# ─────────────────────────────────────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────────────────────────────────────

logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "q-chem.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="500 MB",
)


# ─────────────────────────────────────────────────────────────────────────────
# ReAct prompt
# ─────────────────────────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
    """You are a quantum chemistry assistant. Help users generate configurations for quantum chemistry calculations.

Available tools:
{tools}

Use this format:
Question: the input question
Thought: what you should do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result
... (repeat as needed)
Thought: I now have the answer
Final Answer: the final response

Past conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent executor
# ─────────────────────────────────────────────────────────────────────────────

tools = get_tools()
executor = AgentExecutor(
    agent=create_react_agent(llm, tools, REACT_PROMPT),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)


# ─────────────────────────────────────────────────────────────────────────────
# Simple conversation memory
# ─────────────────────────────────────────────────────────────────────────────

_memory: dict[str, list[tuple[str, str]]] = {}
"""Session-based memory: {session_id: [(question, answer), ...]}"""


def clear_memory(session_id: str) -> None:
    """Clear memory for a session."""
    if session_id in _memory:
        logger.info(f"Clearing memory for session {session_id}")
        _memory.pop(session_id, None)


def _history_text(session_id: str | None) -> str:
    """Convert past interactions to text for prompt."""
    if not session_id or session_id not in _memory:
        return ""
    
    lines = []
    for q, a in _memory[session_id]:
        lines.append(f"Human: {q}")
        lines.append(f"Assistant: {a}\n")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main agent function
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(query: str, session_id: str | None = None) -> str:
    """Run one step of the agent.
    
    Parameters
    ----------
    query : str
        User question or request.
    session_id : str, optional
        Session ID for persistent memory. If None, no memory is kept.
    
    Returns
    -------
    str
        Agent response.
    """
    logger.info(f"Query (session={session_id}): {query[:100]}")
    
    # Setup Langfuse tracing if configured
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}
    
    # Run agent
    try:
        result = executor.invoke(
            {
                "input": query,
                "chat_history": _history_text(session_id),
            },
            config=config,
        )
        answer = result["output"]
        logger.debug(f"Agent response: {answer[:200]}")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        answer = f"Error: {str(e)}"
    
    # Store in memory
    if session_id:
        _memory.setdefault(session_id, []).append((query, answer))
        logger.debug(f"Memory updated for {session_id}")
    
    # Flush Langfuse if available
    if handler:
        try:
            from langfuse import Langfuse
            Langfuse().flush()
        except Exception as e:
            logger.warning(f"Langfuse flush error: {e}")
    
    return answer


__all__ = ["run_agent", "clear_memory", "logger"]
