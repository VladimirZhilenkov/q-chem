"""Interactive REPL for local debugging.

REPL (Read-Eval-Print Loop):
  1. Read user input
  2. Evaluate (run agent)
  3. Print result
  4. Repeat

Sessions maintain memory of the conversation.

Usage:
    python -m agent.repl
"""

import uuid
from agent.graph import clear_memory, run_agent, logger


def repl() -> None:
    """Run interactive REPL loop."""
    session_id = str(uuid.uuid4())
    logger.info(f"REPL started with session {session_id}")
    
    print("=" * 70)
    print("Quantum Chemistry Agent - REPL")
    print("=" * 70)
    print("Commands: /help, /clear, /quit")
    print()
    
    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info(f"REPL ended (session {session_id})")
            break
        
        if not query:
            continue
        
        # Handle commands
        if query.lower() in {"/quit", "exit", "quit"}:
            print("Goodbye!")
            logger.info(f"REPL ended (session {session_id})")
            break
        
        if query.lower() == "/clear":
            clear_memory(session_id)
            print("✓ Memory cleared\n")
            continue
        
        if query.lower() == "/help":
            print("""
Available commands:
  /clear  — Clear conversation history
  /quit   — Exit REPL
  /help   — Show this help

Ask me to:
  • Generate quantum chemistry configs for PySCF, ORCA, Psi4, xTB
  • Parse molecule descriptions
  • Validate basis sets
  • Fetch molecules from PubChem
""")
            continue
        
        # Run agent
        try:
            response = run_agent(query, session_id=session_id)
            print(f"\nAgent> {response}\n")
        except Exception as exc:
            logger.error(f"REPL error: {exc}")
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    repl()
