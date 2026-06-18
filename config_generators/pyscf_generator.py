"""PySCF Python script generator from QChemJob."""

from schemas import QChemJob


def generate_pyscf(job: QChemJob) -> str:
    """Generate PySCF Python script from QChemJob specification.
    
    Parameters
    ----------
    job : QChemJob
        Quantum chemistry job specification.
        
    Returns
    -------
    str
        Executable Python script using PySCF.
    """
    # Build molecule geometry string
    geom_lines = []
    for symbol, x, y, z in job.atoms:
        geom_lines.append(f"  {symbol:2} {x:12.6f} {y:12.6f} {z:12.6f}")
    geom_str = "\n".join(geom_lines)
    
    # Prepare method string (lowercase for PySCF)
    method = job.method.lower()
    
    # Prepare basis (remove spaces for PySCF)
    basis = job.basis.replace(" ", "-")
    
    # PCM/CPCM solvent support
    solvent_code = ""
    if job.solvent:
        solvent_code = f"""
# Apply CPCM solvent
from pyscf import solvent
mf = solvent.CPCM(mf)
mf.with_solvent.solvent = '{job.solvent}'
"""
    
    script = f"""#!/usr/bin/env python
\"\"\"PySCF calculation: {job.id}
Method: {job.method} / {job.basis}
Job type: {job.job_type}
\"\"\"

import sys
from pyscf import gto, scf, mp, cc, dft

# Build molecule
mol = gto.M(
    atom='''
{geom_str}
    ''',
    basis='{basis}',
    charge={job.charge},
    spin={job.multiplicity - 1},
)

# Prepare molecule
mol.verbose = 4
mol.build()

# Method dispatch
method = '{method}'

if method in ['hf', 'rhf']:
    mf = scf.RHF(mol)
elif method in ['uhf']:
    mf = scf.UHF(mol)
elif method.startswith('b3lyp') or method.startswith('dft') or '/' in method:
    # DFT method
    xc = '{job.method}'
    mf = dft.RKS(mol, xc=xc)
elif method in ['mp2']:
    rhf = scf.RHF(mol)
    rhf.verbose = 4
    rhf.kernel()
    mf = mp.MP2(rhf)
elif method.startswith('ccsd'):
    rhf = scf.RHF(mol)
    rhf.verbose = 4
    rhf.kernel()
    mf = cc.CCSD(rhf)
else:
    raise ValueError(f"Unknown method: {{method}}")

{solvent_code}

# Run calculation
mf.verbose = 4
if '{job.job_type}' == 'opt':
    from pyscf.geomopt.geometric_solver import optimize
    mol_opt = optimize(mf)
    energy = mf.e_tot
elif '{job.job_type}' == 'freq':
    from pyscf import hessian
    mf.kernel()
    energy = mf.e_tot
    h = mf.Hessian().kernel()
else:
    energy = mf.kernel()

# Print results
print(f"Energy (Hartree): {{energy:.10f}}")

try:
    dipole = mf.dip_moment(mol)
    print(f"Dipole (Debye): {{dipole}}")
except:
    pass

converged = getattr(mf, 'converged', True)
print(f"Converged: {{converged}}")
"""
    
    return script

