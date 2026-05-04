"""Prompt-based ReAct agent (works with LLMs that lack native tool-calling)."""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from agent.llm import get_langfuse_handler, llm
from agent.registry import get_tools

REACT_PROMPT = PromptTemplate.from_template(
    """You are a quantum chemistry assistant.

CRITICAL RULES:
1. standardize_chem_input returns a human-readable TEXT summary (not an object).
   Example output: "Molecule: 2 atoms, method: hf, basis: sto-3g"
   DO NOT try to parse it or pass it to other tools.

2. run_pyscf expects RAW ORCA or Psi4 INPUT STRING (the original input with ! or molecule).
   NEVER pass the output of standardize_chem_input to run_pyscf.
   If you need to run a calculation, use the original input from the user.

3. If user asks ONLY to parse ("распарси"), call standardize_chem_input and STOP.
   Do NOT call run_pyscf.

4. If user asks to calculate ("посчитай", "calculate"), call run_pyscf with the ORIGINAL input string.

5. NEVER invent or simulate results. Only return what tools actually return.

Rules for chemistry:
- Double-hybrid DFT (B2PLYP, PWPB95, DSD-*) must be detected BEFORE general DFT
  and requires explicit xc-string + scaled MP2 on KS orbitals.
- For SOS-functionals, same-spin scaling = 0.0.
- Always report units (Hartree, Debye, cm^-1) in the final answer.

You have access to the following tools:

{tools}

Use EXACTLY this format:

Question: the input question
Thought: reasoning about what to do
Action: one of [{tool_names}]
Action Input: a JSON object with tool arguments
Observation: the tool's output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the user

Question: {input}
Thought:{agent_scratchpad}"""
)

tools = get_tools()
_react_agent = create_react_agent(llm, tools, REACT_PROMPT)
executor = AgentExecutor(
    agent=_react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)


def _build_config(session_id: str | None):
    handler = get_langfuse_handler()
    config: dict = {"callbacks": [handler]} if handler else {}
    if session_id:
        config.setdefault("metadata", {})["session_id"] = session_id
    return config, handler


def _flush(handler):
    if handler is None:
        return
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass


def run_agent(query: str, session_id: str | None = None) -> str:
    """Invoke the ReAct agent; returns final answer string."""
    config, handler = _build_config(session_id)
    try:
        result = executor.invoke({"input": query}, config=config)
        return result["output"]
    finally:
        _flush(handler)


def stream_agent(query: str, session_id: str | None = None):
    """Stream intermediate steps."""
    config, handler = _build_config(session_id)
    try:
        for step in executor.stream({"input": query}, config=config):
            yield step
    finally:
        _flush(handler)


__all__ = ["executor", "run_agent", "stream_agent"]
