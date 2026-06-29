"""PySCF Python script generator from QChemJob."""

from schemas import QChemJob


# Static dielectric constants (eps, 25 °C) for common solvents. PySCF's PCM model
# is parameterised by this number rather than by a solvent name; values match the
# tables used by ORCA/Gaussian CPCM. Keys are lower-cased with common aliases.
SOLVENT_DIELECTRIC = {
    "water": 78.3553, "h2o": 78.3553,
    "methanol": 32.613, "meoh": 32.613,
    "ethanol": 24.852, "etoh": 24.852,
    "acetonitrile": 35.688, "mecn": 35.688, "acn": 35.688,
    "dmso": 46.826, "dimethylsulfoxide": 46.826,
    "dmf": 37.219, "n,n-dimethylformamide": 37.219,
    "acetone": 20.493,
    "dichloromethane": 8.93, "dcm": 8.93, "methylenechloride": 8.93,
    "chloroform": 4.7113, "chcl3": 4.7113,
    "thf": 7.4257, "tetrahydrofuran": 7.4257,
    "toluene": 2.3741,
    "benzene": 2.2706,
    "hexane": 1.8819, "n-hexane": 1.8819,
    "diethylether": 4.2400, "ether": 4.2400, "et2o": 4.2400,
    "ethylacetate": 5.9867, "ethyl-acetate": 5.9867,
    "dioxane": 2.2099, "1,4-dioxane": 2.2099,
    "pyridine": 12.978,
    "carbontetrachloride": 2.2280, "ccl4": 2.2280,
    "nitromethane": 36.562,
    "aceticacid": 6.2528, "acetic-acid": 6.2528,
    "ammonia": 22.5,
}


def _solvent_eps(name: str) -> float:
    """Look up the dielectric constant for a solvent name, or raise with hints."""
    key = name.strip().lower().replace(" ", "")
    if key in SOLVENT_DIELECTRIC:
        return SOLVENT_DIELECTRIC[key]
    raise ValueError(
        f"Unknown solvent '{name}'. Supported solvents: "
        f"{', '.join(sorted(SOLVENT_DIELECTRIC))}."
    )


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
    
    # Implicit-solvent support.
    # PySCF's solvent module no longer exposes a `CPCM` constructor. We use the
    # PCM family (solvent.PCM) with the C-PCM variant, which is parameterised by a
    # numeric dielectric `eps` (not a solvent name), so the job's solvent name is
    # resolved to its dielectric constant here at generation time.
    solvent_code = ""
    if job.solvent:
        eps = _solvent_eps(job.solvent)
        solvent_code = f"""
# Apply C-PCM implicit solvent ({job.solvent}, eps={eps})
from pyscf import solvent  # noqa: F401
mf = solvent.PCM(mf)
mf.with_solvent.method = 'C-PCM'
mf.with_solvent.eps = {eps}
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
import re as _re
method = '{method}'
open_shell = mol.spin != 0   # restricted for closed shells, unrestricted otherwise

# Split an optional empirical-dispersion suffix (-d3, -d3bj, -d4) off the
# functional name, e.g. 'b3lyp-d3bj' -> xc='b3lyp', disp='d3bj'.
_disp = None
_dm = _re.match(r'^(.*?)-(d3bj|d3zero|d3|d4)$', method)
if _dm:
    xc_name, _disp = _dm.group(1), _dm.group(2)
    if _disp == 'd3':
        _disp = 'd3zero'   # bare '-d3' means zero-damping
else:
    xc_name = method

# Known wavefunction methods are matched explicitly; ANY other name is treated
# as a DFT exchange-correlation functional and handed straight to libxc
# (b3lyp, pbe0, pbe, m06-2x, tpss, ...). Reference (R/U) follows spin.
if method in ('hf', 'rhf', 'uhf', 'rohf'):
    mf = scf.UHF(mol) if open_shell else scf.RHF(mol)
elif method == 'mp2':
    ref = scf.UHF(mol) if open_shell else scf.RHF(mol)
    ref.verbose = 4
    ref.kernel()
    mf = mp.MP2(ref)
elif method.startswith('ccsd'):
    ref = scf.UHF(mol) if open_shell else scf.RHF(mol)
    ref.verbose = 4
    ref.kernel()
    mf = cc.CCSD(ref)
elif xc_name.startswith('gfn') or xc_name in ('gfnff', 'ipea'):
    # Semiempirical tight-binding methods are NOT DFT functionals and libxc
    # does not know them — feeding 'gfn2' to dft.RKS yields the cryptic
    # "LibXCFunctional: name 'GFN2' not found". Fail with a clear pointer to xTB.
    raise RuntimeError(
        f"'{{method}}' is a semiempirical (xTB) method, not a PySCF/DFT functional. "
        "Run it with the xTB engine instead (engine='xtb')."
    )
else:
    # DFT functional — PySCF/libxc validates the name and raises if unknown
    mf = dft.UKS(mol, xc=xc_name) if open_shell else dft.RKS(mol, xc=xc_name)
    if _disp:
        # D3/D4 need the dftd3/dftd4 native backend (pyscf.dispersion). It is a
        # system-level dependency (best installed via conda-forge); fail with a
        # clear, actionable message rather than a cryptic loader traceback.
        try:
            from pyscf.dispersion import dftd3, dftd4   # noqa: F401
        except Exception:
            raise RuntimeError(
                f"Dispersion '{{_disp}}' requested for functional '{{xc_name}}', but the "
                "dftd3/dftd4 backend is not installed. Install it via conda-forge "
                "(conda install -c conda-forge dftd3-python dftd4-python), or rerun "
                f"with the plain functional (e.g. '{{xc_name}}') without the dispersion suffix."
            )
        mf.disp = _disp

{solvent_code}

# Run calculation
mf.verbose = 4
job_type = '{job.job_type}'

if job_type == 'opt':
    from pyscf.geomopt.geometric_solver import optimize
    mol_eq = optimize(mf)
    mf.reset(mol_eq)              # rebind to optimized geometry, keep method/xc
    energy = mf.kernel()
    print("=== OPTIMIZED GEOMETRY (Angstrom) ===")
    coords = mol_eq.atom_coords(unit='Angstrom')
    for i in range(mol_eq.natm):
        sym = mol_eq.atom_symbol(i)
        x, y, z = coords[i]
        print(f"{{sym}} {{x:.6f}} {{y:.6f}} {{z:.6f}}")
    print("=== END GEOMETRY ===")
elif job_type == 'freq':
    from pyscf.hessian import thermo
    energy = mf.kernel()
    hess = mf.Hessian().kernel()
    ha = thermo.harmonic_analysis(mf.mol, hess)
    print("=== VIBRATIONAL FREQUENCIES (cm^-1) ===")
    for i, w in enumerate(ha['freq_wavenumber'], 1):
        w = complex(w)
        # imaginary modes (transition states) are reported as negative wavenumbers
        val = w.real if abs(w.imag) < 1e-3 else -abs(w.imag)
        print(f"Frequency {{i}}: {{val:.4f}}")
    th = thermo.thermo(mf, ha['freq_au'], 298.15, 101325)
    print("=== THERMOCHEMISTRY (298.15 K) ===")
    print(f"ZPE (Hartree): {{th['ZPE'][0]:.8f}}")
    print(f"Enthalpy (Hartree): {{th['H_tot'][0]:.8f}}")
    print(f"Gibbs (Hartree): {{th['G_tot'][0]:.8f}}")
    print(f"Entropy (Hartree/K): {{th['S_tot'][0]:.10f}}")
else:
    energy = mf.kernel()

# Print results
print(f"Energy (Hartree): {{energy:.10f}}")

try:
    dipole = mf.dip_moment(mf.mol)
    print(f"Dipole (Debye): {{dipole}}")
except Exception:
    pass

# Frontier orbitals (HOMO/LUMO) and gap — works for HF/DFT (RKS/UKS)
try:
    import numpy as _np
    _mo_e = _np.asarray(mf.mo_energy)
    _mo_occ = _np.asarray(mf.mo_occ)
    _occ = _mo_e[_mo_occ > 0]
    _vir = _mo_e[_mo_occ == 0]
    if _occ.size and _vir.size:
        _homo = float(_occ.max())
        _lumo = float(_vir.min())
        print(f"HOMO (Hartree): {{_homo:.6f}}")
        print(f"LUMO (Hartree): {{_lumo:.6f}}")
        print(f"HOMO-LUMO gap (eV): {{(_lumo - _homo) * 27.211386245988:.4f}}")
except Exception:
    pass

# Mulliken partial charges
try:
    _pop, _charges = mf.mulliken_pop(verbose=0)
    print("=== MULLIKEN CHARGES ===")
    for _i in range(mf.mol.natm):
        print(f"{{mf.mol.atom_symbol(_i)}} {{float(_charges[_i]):.6f}}")
    print("=== END MULLIKEN ===")
except Exception:
    pass

converged = getattr(mf, 'converged', True)
print(f"Converged: {{converged}}")
"""
    
    return script

