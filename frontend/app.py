"""Chainlit UI for the quantum chemistry agent."""

import logging
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

logging.getLogger("engineio.server").setLevel(logging.CRITICAL)
logging.getLogger("socketio.server").setLevel(logging.CRITICAL)

import chainlit as cl

from agent.graph import clear_memory, run_agent

SESSION_KEY = "agent_session_id"


@cl.on_chat_start
async def start():
    cl.user_session.set(SESSION_KEY, str(uuid.uuid4()))
    await cl.Message(
        content=(
            "Quantum Chemistry Assistant\n\n"
            "Ask about molecules, ORCA/Psi4 input, or calculations.\n"
            "Dialogue history is kept for this chat session."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get(SESSION_KEY)
    msg = cl.Message(content="")
    await msg.send()

    try:
        msg.content = run_agent(message.content, session_id=session_id)
        await msg.update()
    except Exception as exc:
        msg.content = f"Error: {exc}"
        await msg.update()


@cl.on_chat_end
async def end():
    session_id = cl.user_session.get(SESSION_KEY)
    if session_id:
        clear_memory(session_id)
