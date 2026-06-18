"""xTB input file generator from QChemJob."""

from schemas import QChemJob


def generate_xtb(job: QChemJob) -> str:
    """Generate xTB control file and XYZ structure from QChemJob.
    
    Parameters
    ----------
    job : QChemJob
        Quantum chemistry job specification.
        
    Returns
    -------
    str
        Combined XYZ structure and xtb control file in one string.
    """
    # Generate XYZ format
    xyz_lines = [str(len(job.atoms))]
    xyz_lines.append(f"xTB calculation: {job.id}")
    
    for symbol, x, y, z in job.atoms:
        xyz_lines.append(f"{symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}")
    
    xyz_str = "\n".join(xyz_lines)
    
    # Map method
    method_map = {
        "gfn2": "gfn2-xtb",
        "gfn1": "gfn1-xtb",
        "ipea": "ipea-xtb",
    }
    method = method_map.get(job.method.lower(), "gfn2-xtb")
    
    # Job type
    job_type_map = {
        "energy": "",
        "opt": "--opt",
        "freq": "--freq",
    }
    job_flag = job_type_map.get(job.job_type, "")
    
    # Control file
    control_file = f"""$control
title {job.id}
method {method}
charge {job.charge}
multiplicity {job.multiplicity}
$end

$cmd
xtb {job_flag} struct.xyz
$end
"""
    
    result = f"""# ===== XYZ Structure =====
{xyz_str}

# ===== xTB Control File =====
{control_file}

# ===== Running xTB =====
# Save XYZ to struct.xyz, then run:
# xtb {job_flag} struct.xyz
"""
    
    return result
