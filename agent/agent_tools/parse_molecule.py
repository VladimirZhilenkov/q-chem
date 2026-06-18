"""Tool: Parse molecule specifications from user input."""

from typing import Annotated
from langchain_core.tools import tool
import re


@tool
def parse_molecule(
    description: Annotated[str, "User description of molecule (e.g., 'water molecule at standard geometry' or 'benzene C6H6')"],
) -> str:
    """Parse natural language molecule description and suggest coordinates.
    
    Returns formatted atom list or explanation of what's needed.
    """
    desc = description.lower().strip()
    
    # Common molecules
    molecules = {
        "water": [("O", 0.0, 0.0, 0.0), ("H", 0.957, 0.0, 0.0), ("H", -0.239, 0.927, 0.0)],
        "h2o": [("O", 0.0, 0.0, 0.0), ("H", 0.957, 0.0, 0.0), ("H", -0.239, 0.927, 0.0)],
        "methane": [("C", 0.0, 0.0, 0.0), ("H", 0.629, 0.629, 0.629), ("H", -0.629, -0.629, 0.629), 
                    ("H", -0.629, 0.629, -0.629), ("H", 0.629, -0.629, -0.629)],
        "ch4": [("C", 0.0, 0.0, 0.0), ("H", 0.629, 0.629, 0.629), ("H", -0.629, -0.629, 0.629),
                ("H", -0.629, 0.629, -0.629), ("H", 0.629, -0.629, -0.629)],
        "ammonia": [("N", 0.0, 0.0, 0.0), ("H", 0.940, 0.0, 0.0), ("H", -0.470, 0.814, 0.0), ("H", -0.470, -0.814, 0.0)],
        "nh3": [("N", 0.0, 0.0, 0.0), ("H", 0.940, 0.0, 0.0), ("H", -0.470, 0.814, 0.0), ("H", -0.470, -0.814, 0.0)],
        "benzene": [
            ("C", 1.401, 0.0, 0.0), ("C", 0.7005, 1.213, 0.0), ("C", -0.7005, 1.213, 0.0),
            ("C", -1.401, 0.0, 0.0), ("C", -0.7005, -1.213, 0.0), ("C", 0.7005, -1.213, 0.0),
            ("H", 2.49, 0.0, 0.0), ("H", 1.245, 2.157, 0.0), ("H", -1.245, 2.157, 0.0),
            ("H", -2.49, 0.0, 0.0), ("H", -1.245, -2.157, 0.0), ("H", 1.245, -2.157, 0.0),
        ],
        "c6h6": [
            ("C", 1.401, 0.0, 0.0), ("C", 0.7005, 1.213, 0.0), ("C", -0.7005, 1.213, 0.0),
            ("C", -1.401, 0.0, 0.0), ("C", -0.7005, -1.213, 0.0), ("C", 0.7005, -1.213, 0.0),
            ("H", 2.49, 0.0, 0.0), ("H", 1.245, 2.157, 0.0), ("H", -1.245, 2.157, 0.0),
            ("H", -2.49, 0.0, 0.0), ("H", -1.245, -2.157, 0.0), ("H", 1.245, -2.157, 0.0),
        ],
        "ethane": [
            ("C", 0.0, 0.0, 0.0), ("C", 1.536, 0.0, 0.0),
            ("H", -0.512, 0.887, 0.0), ("H", -0.512, -0.444, 0.768), ("H", -0.512, -0.444, -0.768),
            ("H", 2.048, 0.887, 0.0), ("H", 2.048, -0.444, 0.768), ("H", 2.048, -0.444, -0.768),
        ],
        "h2": [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
        "n2": [("N", 0.0, 0.0, 0.0), ("N", 1.097, 0.0, 0.0)],
        "o2": [("O", 0.0, 0.0, 0.0), ("O", 1.207, 0.0, 0.0)],
    }
    
    # Check for molecule name
    for mol_name, coords in molecules.items():
        if mol_name in desc:
            atom_str = ", ".join(f"{sym} {x:.3f} {y:.3f} {z:.3f}" for sym, x, y, z in coords)
            return f"Recognized molecule: {mol_name.upper()}\n\nAtoms:\n{atom_str}"
    
    # Try to extract elements and formula
    elements_pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(elements_pattern, description)
    if matches:
        formula = "".join(f"{e}{n or '1'}" for e, n in matches)
        return f"Found formula: {formula}\n\nPlease provide specific coordinates or use a recognized molecule name (water, methane, benzene, etc.)"
    
    return "Could not parse molecule. Please provide: 1) recognized name (water, benzene, etc.) or 2) coordinates in format 'SYMBOL X Y Z'"
