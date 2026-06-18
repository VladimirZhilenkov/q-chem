"""Config generators for quantum chemistry calculations.

Separate modules for each software package (PySCF, ORCA, Psi4, xTB).
Each generator takes a QChemJob and produces formatted input file text.
"""

from .pyscf_generator import generate_pyscf
from .orca_generator import generate_orca
from .psi4_generator import generate_psi4
from .xtb_generator import generate_xtb

__all__ = [
    "generate_pyscf",
    "generate_orca", 
    "generate_psi4",
    "generate_xtb",
]
