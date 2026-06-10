from typing import Annotated

from langchain_core.tools import tool


@tool
def echo(message: Annotated[str, "Text to echo back."]) -> str:
    """Echo the input message."""
    return f"Echo response: {message}"
