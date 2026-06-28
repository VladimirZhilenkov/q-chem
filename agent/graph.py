"""Tool calling agent with simple dialogue memory and logging.

Architecture:
  1. PROMPT     — ChatPromptTemplate with MessagesPlaceholder
  2. agent      — create_tool_calling_agent (native tool calling)
  3. executor   — tool calling loop (not ReAct format)
  4. _memory    — conversation history by session_id
  5. run_agent  — main entry point for UI and REPL
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
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
# Tool calling prompt (native format, no JSON instructions)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quantum chemistry assistant. Help users generate configurations for quantum chemistry calculations.

You have access to tools that can:
• Parse molecule descriptions and recognize common molecules (water, methane, benzene, etc.)
• Validate basis sets in the Basis Set Exchange
• Generate quantum chemistry input files for PySCF, ORCA, Psi4, or xTB
• Fetch molecular structures from PubChem
• Echo/debug messages

Use the available tools to help the user. Call tools when needed to complete the task.
Be concise and direct in your responses.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────────────────────────────────────

tools = get_tools()

# Create agent with native tool calling (not ReAct format)
agent = create_tool_calling_agent(llm, tools, TOOL_CALLING_PROMPT)

# Agent executor with same parameters as before
executor = AgentExecutor(
    agent=agent,
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


def _history_messages(session_id: str | None) -> list:
    """Convert stored history to list of BaseMessage objects for chat_history.
    
    Parameters
    ----------
    session_id : str | None
        Session ID to retrieve history from.
    
    Returns
    -------
    list
        List of HumanMessage and AIMessage objects.
    """
    if not session_id or session_id not in _memory:
        return []
    
    messages = []
    for question, answer in _memory[session_id]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Main agent function
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(query: str, session_id: str | None = None) -> str:
    """Run one step of the agent with tool calling.
    
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
    
    # Run agent with message-based history (tool calling format)
    try:
        result = executor.invoke(
            {
                "input": query,
                "chat_history": _history_messages(session_id),
            },
            config=config,
        )
        answer = result["output"]
        logger.debug(f"Agent response: {answer[:200]}")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        answer = f"Error: {str(e)}"
    
    # Store in memory (same format: (question, answer) tuples)
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
