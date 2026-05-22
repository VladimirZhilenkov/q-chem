import sys
import importlib

# Try different import paths
paths = [
    "langchain.agents.agent",
    "langchain.agents.executor",
    "langchain_community.agents.agent",
    "langchain_community.agents.executor",
    "langchain.agents.tools",
]

for path in paths:
    try:
        mod = importlib.import_module(path)
        if hasattr(mod, 'AgentExecutor'):
            print(f"Found AgentExecutor in {path}")
            sys.exit(0)
    except Exception as e:
        pass

# If not found, let's see what's actually available
try:
    from langchain import agents
    print("Available in langchain.agents:")
    print([x for x in dir(agents) if not x.startswith('_')])
except Exception as e:
    print(f"Error: {e}")
