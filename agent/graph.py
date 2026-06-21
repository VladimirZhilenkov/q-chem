"""Structured-chat agent with session memory and Langfuse tracing.

Architecture:
  1. PROMPT    — system instructions + chat history + scratchpad placeholders
  2. executor  — structured-chat loop (JSON action blobs over plain text)
  3. _memory   — capped conversation history by session_id
  4. run_agent — main entry point for UI and REPL

Uses create_structured_chat_agent (not the legacy single-string ReAct agent,
and not native tool-calling). The model emits a JSON blob with `action` and a
dict `action_input`, so multi-argument tools like generate_config receive each
argument correctly — and it works on OpenAI-compatible endpoints that do NOT
have server-side tool/function calling enabled.
"""

import sys

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger

from agent.llm import flush_langfuse, get_langfuse_handler, langfuse_trace, llm
from agent.registry import get_tools


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logger.remove()
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
# Structured-chat prompt
# ─────────────────────────────────────────────────────────────────────────────
#
# The model never sees native tool/function-calling. Instead it is instructed to
# emit a JSON action blob. {tools} and {tool_names} are filled by
# create_structured_chat_agent; {input}, {agent_scratchpad}, {chat_history} are
# provided per-invocation.

_SYSTEM = """You are a quantum chemistry assistant. Help users generate configurations for quantum chemistry calculations.

When you need a molecule's geometry, first try parse_molecule for common molecules, and fall back to get_molecule_from_pubchem for anything else. Use validate_basis to confirm a basis set before generating a config, and generate_config to produce the final input file. If the user asks you to RUN or EXECUTE the calculation, call run_calculation with ORCA-format input (generate it with generate_config using engine='orca', or use the ORCA block from get_molecule_from_pubchem) — it runs locally via PySCF and returns the energy and other results.

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an "action" key (tool name) and an "action_input" key (a dict of the tool's arguments).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per JSON blob, like this:

```
{{
  "action": $TOOL_NAME,
  "action_input": {{"arg1": "value1", "arg2": "value2"}}
}}
```

Follow this format:

Question: the input question to answer
Thought: consider what to do next
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Action:
```
{{
  "action": "Final Answer",
  "action_input": "the final response to the human"
}}
```

Always pass ALL required arguments for a tool inside "action_input" in a single action. Begin! Always respond with a single valid JSON blob for the action."""

_HUMAN = """{input}

{agent_scratchpad}
(reminder to respond with a single JSON blob for one action)"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", _HUMAN),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent executor
# ─────────────────────────────────────────────────────────────────────────────

tools = get_tools()
executor = AgentExecutor(
    agent=create_structured_chat_agent(llm, tools, PROMPT),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session memory
# ─────────────────────────────────────────────────────────────────────────────

_memory: dict[str, list[tuple[str, str]]] = {}
_MAX_HISTORY = 10  # max exchanges kept per session


def clear_memory(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in _memory:
        logger.info(f"Clearing memory for session {session_id}")
        _memory.pop(session_id, None)


def _history_messages(session_id: str | None) -> list[BaseMessage]:
    """Build chat history as alternating Human/AI messages for the prompt."""
    if not session_id or session_id not in _memory:
        return []
    messages: list[BaseMessage] = []
    for q, a in _memory[session_id]:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))
    return messages


def _store_memory(session_id: str, query: str, answer: str) -> None:
    history = _memory.setdefault(session_id, [])
    history.append((query, answer))
    # Keep only the most recent exchanges to avoid unbounded growth
    if len(history) > _MAX_HISTORY:
        _memory[session_id] = history[-_MAX_HISTORY:]
    logger.debug(f"Memory updated for {session_id} ({len(_memory[session_id])} turns)")


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
        Session ID for persistent memory and Langfuse trace grouping.

    Returns
    -------
    str
        Agent response, or an error message prefixed with "Error:".
    """
    logger.info(f"Query (session={session_id}): {query[:100]}")

    handler = get_langfuse_handler(session_id)
    config = {"callbacks": [handler]} if handler else {}

    with langfuse_trace(session_id):
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
            # Only store successful responses in memory
            if session_id:
                _store_memory(session_id, query, answer)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            answer = f"Error: {e}"

    # Flush outside the trace context so all spans are finalized first
    flush_langfuse(handler)
    return answer


__all__ = ["run_agent", "clear_memory", "logger"]
