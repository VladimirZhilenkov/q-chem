"""ORCA input file generator from QChemJob."""

from schemas import QChemJob


def generate_orca(job: QChemJob) -> str:
    """Generate ORCA input file from QChemJob specification.
    
    Parameters
    ----------
    job : QChemJob
        Quantum chemistry job specification.
        
    Returns
    -------
    str
        ORCA input file text.
    """
    # Build geometry block
    geom_lines = []
    for symbol, x, y, z in job.atoms:
        geom_lines.append(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}")
    geom_str = "\n".join(geom_lines)
    
    # Map job type to ORCA keyword
    job_type_map = {
        "energy": "SP",
        "opt": "Opt",
        "freq": "Freq",
    }
    job_keyword = job_type_map.get(job.job_type, "SP")
    
    # Method
    method = job.method.upper()
    
    # Basis (remove spaces for ORCA notation)
    basis = job.basis.upper().replace(" ", "")
    
    # Build route line
    route = f"! {method} {basis} {job_keyword}"
    
    # Add solvent if specified
    if job.solvent:
        route += f" CPCM({job.solvent})"
    
    # Multiplicity
    mult = job.multiplicity
    
    # Charge
    charge = job.charge
    
    input_text = f"""{route}

* xyz {charge} {mult}
{geom_str}
*
"""
    
    return input_text
