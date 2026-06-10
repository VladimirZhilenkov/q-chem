"""ReAct-агент с диалоговой памятью.

Структура (читать сверху вниз):
  1. PROMPT   — как LLM рассуждает и вызывает tools
  2. executor — цикл Thought → Action → Observation
  3. _memory  — история чата по session_id
  4. run_agent — одна точка входа для UI и REPL
"""

import uuid

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from agent.llm import get_langfuse_handler, llm
from agent.registry import get_tools


REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{chat_history}Question: {input}
Thought:{agent_scratchpad}"""
)


tools = get_tools()
executor = AgentExecutor(
    agent=create_react_agent(llm, tools, REACT_PROMPT),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)


_memory: dict[str, list[tuple[str, str]]] = {}


def clear_memory(session_id: str) -> None:
    _memory.pop(session_id, None)


def _history_text(session_id: str | None) -> str:
    """Собрать прошлые реплики в строку для промпта."""
    if not session_id or session_id not in _memory:
        return ""
    lines = []
    for question, answer in _memory[session_id]:
        lines.append(f"Human: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines) + "\n"


def run_agent(query: str, session_id: str | None = None) -> str:
    """Один ход агента. Память работает только если передан session_id."""
    # Опциональный трейсинг в Langfuse (если настроен .env)
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}

    result = executor.invoke(
        {"input": query, "chat_history": _history_text(session_id)},
        config=config,
    )
    answer = result["output"]

    if session_id:
        _memory.setdefault(session_id, []).append((query, answer))

    if handler:
        try:
            from langfuse import Langfuse
            Langfuse().flush()
        except Exception:
            pass

    return answer


def repl() -> None:
    """Терминальный чат для локальной отладки."""
    session_id = str(uuid.uuid4())
    print("REPL запущен. Команды: exit, quit, /clear\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "/clear":
            clear_memory(session_id)
            print("Память очищена.\n")
            continue

        try:
            print(f"\nAgent> {run_agent(query, session_id=session_id)}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    repl()
