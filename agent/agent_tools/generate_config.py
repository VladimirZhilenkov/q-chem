"""Tool: Generate quantum chemistry configuration files."""

from typing import Annotated
from langchain_core.tools import tool
from schemas import QChemJob
from config_generators import generate_pyscf, generate_orca, generate_psi4, generate_xtb
import json


@tool
def generate_config(
    atoms: Annotated[str, "List of atoms as 'SYMBOL X Y Z' lines, comma-separated"],
    method: Annotated[str, "Calculation method (e.g., B3LYP, HF, CCSD)"],
    basis: Annotated[str, "Basis set name (e.g., def2-TZVP, 6-31G)"],
    charge: Annotated[int, "Molecular charge"],
    multiplicity: Annotated[int, "Spin multiplicity (1=singlet, 2=doublet, etc.)"],
    job_type: Annotated[str, "Job type: energy, opt, or freq"] = "energy",
    engine: Annotated[str, "Engine: pyscf, orca, psi4, or xtb"] = "pyscf",
    solvent: Annotated[str, "Solvent name or None"] = None,
) -> str:
    """Generate a quantum chemistry config file for the specified engine.
    
    Returns formatted config text ready to use.
    """
    # Parse atoms
    atom_list = []
    for line in atoms.split(","):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atom_list.append((sym, x, y, z))
            except ValueError:
                return f"Error parsing atoms: invalid coordinates in '{line}'"
    
    if not atom_list:
        return "Error: no atoms provided"
    
    # Create QChemJob
    job = QChemJob(
        id="generated_job",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        atoms=atom_list,
        job_type=job_type,
        engine=engine,
        solvent=solvent if solvent and solvent != "None" else None,
    )
    
    # Generate config
    if engine == "pyscf":
        config = generate_pyscf(job)
    elif engine == "orca":
        config = generate_orca(job)
    elif engine == "psi4":
        config = generate_psi4(job)
    elif engine == "xtb":
        config = generate_xtb(job)
    else:
        return f"Unknown engine: {engine}"
    
    return config
