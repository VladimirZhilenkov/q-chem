"""Registry of tools available to the agent."""

from .agent_tools.debug import echo
from .agent_tools.pubchem import get_molecule_from_pubchem
from .agent_tools.generate_config import generate_config
from .agent_tools.parse_molecule import parse_molecule
from .agent_tools.validate_basis import validate_basis

TOOLS = [
    parse_molecule,
    get_molecule_from_pubchem,
    validate_basis,
    generate_config,
    echo,
]


def get_tools():
    """Return list of tools available to the agent."""
    return TOOLS


__all__ = ["TOOLS", "get_tools"]
