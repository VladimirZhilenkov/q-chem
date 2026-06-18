"""Tool: Validate basis sets."""

from typing import Annotated
from langchain_core.tools import tool
from converter.basis_utils import check_existence


@tool
def validate_basis(
    basis_name: Annotated[str, "Basis set name (e.g., def2-TZVP, 6-31G, cc-pVDZ)"],
) -> str:
    """Check if a basis set exists in the Basis Set Exchange.
    
    Returns confirmation or list of similar basis sets.
    """
    basis_clean = basis_name.strip().lower()
    
    if check_existence(basis_clean):
        return f"✓ Basis set '{basis_name}' is available"
    
    # Suggest alternatives
    suggestions = []
    common_basis = [
        "sto-3g", "3-21g", "6-31g", "6-31g*", "6-31g**", "6-31+g*",
        "6-311g", "6-311g**", "6-311+g**",
        "def2-svp", "def2-tzvp", "def2-tzvpp", "def2-qzvpp",
        "cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z",
        "aug-cc-pvdz", "aug-cc-pvtz",
    ]
    
    for b in common_basis:
        if check_existence(b):
            suggestions.append(b)
    
    result = f"✗ Basis set '{basis_name}' not found in Basis Set Exchange\n\n"
    if suggestions:
        result += f"Available alternatives:\n" + "\n".join(f"  • {b}" for b in suggestions[:10])
    
    return result
