"""Chainlit UI for quantum chemistry configuration agent.

Features:
  • Generate configs for PySCF, ORCA, Psi4, xTB
  • Parse molecule specifications
  • Validate basis sets
  • Fetch molecules from PubChem
  • Persistent conversation memory per session
"""

import logging
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Suppress noisy logs
logging.getLogger("engineio.server").setLevel(logging.CRITICAL)
logging.getLogger("socketio.server").setLevel(logging.CRITICAL)

import chainlit as cl
from agent.graph import clear_memory, run_agent, logger

SESSION_KEY = "agent_session_id"


@cl.on_chat_start
async def start():
    """Initialize session and send welcome message."""
    session_id = str(uuid.uuid4())
    cl.user_session.set(SESSION_KEY, session_id)
    logger.info(f"Chat started: session {session_id}")
    
    await cl.Message(
        content="""**Quantum Chemistry Configuration Agent**

I can help you generate input files for quantum chemistry calculations!

What I can do:
• **Generate configs** for PySCF, ORCA, Psi4, xTB
  - Just describe your molecule and desired calculation
  - Example: "Generate ORCA config for benzene with B3LYP/def2-TZVP energy calculation"
  
• **Parse molecules** - Common molecules are built-in (water, methane, benzene)
• **Validate basis sets** - Check if a basis set is available
• **Fetch from PubChem** - Get molecule structures by name

**Example requests:**
- "Generate PySCF config for water molecule with HF/6-31G"
- "Create ORCA input for benzene optimization with B3LYP"
- "Is def2-TZVP a valid basis set?"
- "Get caffeine from PubChem"
"""
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Process user message."""
    session_id = cl.user_session.get(SESSION_KEY)
    logger.debug(f"Message (session {session_id}): {message.content[:100]}")
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        response = run_agent(message.content, session_id=session_id)
        msg.content = response
        await msg.update()
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        msg.content = f"Error: {exc}"
        await msg.update()


@cl.on_chat_end
async def end():
    """Clean up session."""
    session_id = cl.user_session.get(SESSION_KEY)
    if session_id:
        clear_memory(session_id)
        logger.info(f"Chat ended: session {session_id}")

