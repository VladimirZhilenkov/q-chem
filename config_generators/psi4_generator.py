"""Psi4 input file generator from QChemJob."""

from schemas import QChemJob


def generate_psi4(job: QChemJob) -> str:
    """Generate Psi4 input file from QChemJob specification.
    
    Parameters
    ----------
    job : QChemJob
        Quantum chemistry job specification.
        
    Returns
    -------
    str
        Psi4 input file text.
    """
    # Build geometry block
    geom_lines = []
    for symbol, x, y, z in job.atoms:
        geom_lines.append(f"  {symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}")
    geom_str = "\n".join(geom_lines)
    
    # Job type keywords
    job_type_map = {
        "energy": "energy",
        "opt": "optimize",
        "freq": "frequency",
    }
    job_command = job_type_map.get(job.job_type, "energy")
    
    # Method and basis
    method = job.method.lower()
    basis = job.basis.lower().replace(" ", "-")
    
    # PCM options
    pcm_block = ""
    if job.solvent:
        pcm_block = f"""
set {{
  pcm true
  pcm_scf_type total
}}

pcm_solver = psi4.PCM()
pcm_solver.read_options()
"""
    
    input_text = f"""# Psi4 input: {job.id}
# Method: {job.method} / {job.basis}
# Job: {job.job_type}

molecule mol {{
  {job.charge} {job.multiplicity}
{geom_str}
}}

set basis {basis}
set reference rhf
set scf_type df
set mp2_type df

{pcm_block}

{job_command}('{method}')
"""
    
    return input_text
