from langchain.tools import tool
import asyncio 

@tool
def echo(message:str) -> str:
    """Echo the input message back to the user."""
    return f"Echo response: {message}"