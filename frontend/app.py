"""
app.py
------
Chainlit interface for quantum chemistry agent.
"""

import chainlit as cl
from agent.graph import run_agent, stream_agent


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    
    # Get user's text input
    user_query = message.content
    
    # Send a temporary message while processing
    msg = cl.Message(content="")
    await msg.send()
    
    # Run the agent and get the response
    try:
        # Option 1: Simple blocking call
        response = run_agent(user_query)
        msg.content = response
        await msg.update()
        
        # Option 2: Streaming (uncomment to use instead)
        # async for step in stream_agent(user_query):
        #     if "output" in step:
        #         msg.content = step["output"]
        #         await msg.update()
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()


@cl.on_chat_start
async def start():
    """Send welcome message when chat starts."""
    await cl.Message(
        content="Quantum Chemistry Assistant\n\n"
                "I can help you:\n"
                "- Parse ORCA/Psi4 input: `распарси <input>`\n"
                "- Run calculations: `посчитай <input>`\n\n"
                "Send your quantum chemistry input to get started!"
    ).send()
