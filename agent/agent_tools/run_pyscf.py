"""
run_pyscf.py
-----------
LangChain tool: convert ORCA/Psi4 input → PySCF script → execute → parse result.

Takes raw ORCA or Psi4 input as string, converts to PySCF, runs it via subprocess,
and returns QChemResult with energy, converged status, geometry, etc.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from loguru import logger

from schemas import QChemJob, QChemResult
from converter.qchem_converter import detect_format, parse
from config_generators import generate_pyscf

# Allow overriding the Python used to run PySCF scripts.
# Set PYSCF_PYTHON in .env to point at a conda/WSL Python that has pyscf installed.
_PYSCF_PYTHON = os.getenv("PYSCF_PYTHON", sys.executable)


def run_pyscf(
    orca_or_psi4_input: str,
    fmt: str = "auto",
    timeout_seconds: int = 3600,
) -> QChemResult:
    """
    Convert ORCA/Psi4 input to PySCF, execute the generated script, and return results.

    Parameters
    ----------
    orca_or_psi4_input : str
        Raw input string (ORCA or Psi4 format).
    fmt : str
        Input format: 'auto', 'psi4', or 'orca'. Defaults to 'auto' (auto-detect).
    timeout_seconds : int
        Maximum wall time for the calculation in seconds. Defaults to 3600.

    Returns
    -------
    QChemResult
        Parsed result with energy, convergence status, geometry, dipole, etc.

    Raises
    ------
    ValueError
        If input format cannot be auto-detected.
    RuntimeError
        If subprocess execution fails or times out.
    """
    # Auto-detect format if needed
    if fmt == "auto":
        fmt = detect_format(orca_or_psi4_input, "input")
        if fmt == "unknown":
            raise ValueError(
                "Cannot auto-detect input format. "
                "Pass fmt='psi4' or fmt='orca' explicitly."
            )

    # Parse input → QChemJob, then execute
    job = parse(orca_or_psi4_input, fmt=fmt, source_name="input")
    return _execute_pyscf_job(job, timeout_seconds)


def _execute_pyscf_job(job: QChemJob, timeout_seconds: int = 1800) -> QChemResult:
    """Generate a PySCF script from a QChemJob, run it, and return parsed results."""
    start_time = time.time()
    pyscf_script = generate_pyscf(job)

    # Execute the PySCF script in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "calc.py"
        output_path = tmpdir_path / "output.txt"

        # Write the generated script with UTF-8 encoding
        script_path.write_text(pyscf_script, encoding="utf-8")

        # Run the script via subprocess, capturing stdout and stderr
        try:
            result = subprocess.run(
                [_PYSCF_PYTHON, str(script_path)],
                cwd=str(tmpdir_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            wall_time = time.time() - start_time
            raise RuntimeError(
                f"PySCF calculation timed out after {timeout_seconds}s. "
                f"Elapsed: {wall_time:.1f}s"
            ) from e

        raw_output = result.stdout + "\n" + result.stderr
        wall_time = time.time() - start_time

        if result.returncode != 0:
            if "No module named 'pyscf'" in raw_output or "No module named pyscf" in raw_output:
                raise RuntimeError(
                    "PySCF is not installed in this Python environment.\n\n"
                    "PySCF only runs on Linux/macOS. On Windows you must use WSL2:\n"
                    "  1. Open PowerShell as Administrator and run: wsl --install\n"
                    "  2. Restart your PC, then open Ubuntu and run: pip install pyscf\n"
                    "  3. Find the WSL Python path: wsl which python3\n"
                    "  4. Add to .env: PYSCF_PYTHON=<that path>\n"
                    "  See the project README for full setup instructions."
                )
            raise RuntimeError(
                f"PySCF subprocess failed with return code {result.returncode}. "
                f"Output:\n{raw_output.strip()}"
            )

        # Step 4: Parse the output
        parsed = _parse_pyscf_output(raw_output, pyscf_script, tmpdir_path)

        # Step 5: Build and return QChemResult
        qchem_result = QChemResult(
            job_id="pyscf_run",  # Placeholder; typically set by job manager
            energy=parsed.get("energy"),
            converged=parsed.get("converged", False),
            geometry=parsed.get("geometry"),
            frequencies=parsed.get("frequencies"),
            dipole=parsed.get("dipole"),
            zpe=parsed.get("zpe"),
            enthalpy=parsed.get("enthalpy"),
            gibbs=parsed.get("gibbs"),
            entropy=parsed.get("entropy"),
            homo=parsed.get("homo"),
            lumo=parsed.get("lumo"),
            homo_lumo_gap=parsed.get("homo_lumo_gap"),
            mulliken_charges=parsed.get("mulliken_charges"),
            wall_time=wall_time,
            engine="pyscf",
            method=parsed.get("method", "unknown"),
            basis=parsed.get("basis", "unknown"),
            raw_output=raw_output,
            output_path=str(output_path) if output_path.exists() else None,
        )

        return qchem_result


def _parse_pyscf_output(
    output: str, pyscf_script: str, tmpdir: Path  # noqa: ARG001  (reserved for future checkpoint parsing)
) -> dict:
    """
    Parse PySCF output to extract energy, geometry, dipole, frequencies, convergence.

    Parameters
    ----------
    output : str
        Combined stdout + stderr from PySCF execution.
    pyscf_script : str
        The generated PySCF Python script (for context).
    tmpdir : Path
        Temporary directory where the script ran (for checkpoint files).

    Returns
    -------
    dict
        Dictionary with keys: energy, converged, geometry, dipole, frequencies,
        method, basis, and other extracted properties.
    """
    result = {
        "energy": None,
        "converged": False,
        "geometry": None,
        "dipole": None,
        "frequencies": None,
        "zpe": None,
        "enthalpy": None,
        "gibbs": None,
        "entropy": None,
        "homo": None,
        "lumo": None,
        "homo_lumo_gap": None,
        "mulliken_charges": None,
        "method": "unknown",
        "basis": "unknown",
    }

    # ── Extract basis from script ─────────────────────────────────────────
    # The generated script uses: gto.M(... basis = 'sto-3g', ...)
    basis_match = re.search(r"\bbasis\s*=\s*['\"]([^'\"]+)['\"]", pyscf_script)
    if basis_match:
        result["basis"] = basis_match.group(1).lower()

    # ── Extract method from script ────────────────────────────────────────
    # Priority 0: the authoritative runtime variable `method = '...'`. The whole
    # if/elif dispatch (scf.RHF, dft.RKS, ...) is present in the script as text,
    # so matching those would pick the wrong branch — this assignment is the one
    # that actually runs.
    runtime_method = re.search(r"^method\s*=\s*['\"]([^'\"]+)['\"]", pyscf_script, re.MULTILINE)
    xc_match = re.search(r"mf\.xc\s*=\s*['\"]([^'\"]+)['\"]", pyscf_script)
    if runtime_method:
        result["method"] = runtime_method.group(1).lower()
    elif xc_match:
        # DFT functional — mf.xc = 'b3lyp'
        result["method"] = xc_match.group(1).lower()
    else:
        # Priority 2: Post-HF correlation method — corr = mp.MP2(mf) / cc.CCSD(mf)
        corr_match = re.search(
            r"corr\s*=\s*(mp|cc|ci)\.(U?MP2|U?CCSD|U?CISD)\(", pyscf_script
        )
        if corr_match:
            method_name = corr_match.group(2).lower().lstrip("u")
            result["method"] = method_name
        else:
            # Priority 3: HF — mf = scf.RHF(mol) / scf.UHF(mol)
            mf_match = re.search(
                r"mf\s*=\s*(scf|dft)\.(U?RHF|UHF|RKS|UKS)\(", pyscf_script
            )
            if mf_match:
                cls = mf_match.group(2).lower()
                result["method"] = "hf" if "hf" in cls else cls

    # ── Parse energy ──────────────────────────────────────────────────────
    # The generated script prints an authoritative "Energy (Hartree): ..." line
    # AFTER any optimization. It must take priority: during an opt run PySCF emits
    # many "converged SCF energy = ..." lines (one per geometry step), and the
    # first one is the *starting* geometry — not the optimized result.
    energy_patterns = [
        r"Energy \(Hartree\):\s*([-\d.]+)",       # our authoritative final print
        r"Total energy\s*=\s*([-\d.]+)",
        r"converged SCF energy\s*=\s*([-\d.]+)",  # PySCF native SCF line
        r"E\(\w+\)\s*=\s*([-\d.]+)",              # PySCF: E(RHF) = ...
        r"E\s*=\s*([-\d.]+)\s*Hartree",
        r"Final energy:\s*([-\d.]+)",
    ]
    for pattern in energy_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                result["energy"] = float(match.group(1))
                break
            except (ValueError, IndexError):
                pass

    # ── Parse convergence ─────────────────────────────────────────────────
    # PySCF prints "converged SCF energy = ...", NOT "SCF converged"
    converged_patterns = [
        r"converged SCF energy",          # PySCF RHF/RKS/UHF/UKS standard line
        r"converged\s*=\s*True",          # explicit Python flag
        r"Geometry optimization successful",
        r"Converged in \d+ iterations",
    ]
    for pattern in converged_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            result["converged"] = True
            break

    # Override only on definitive failure indicators (avoid false positives from
    # general "Error" strings that appear in normal PySCF diagnostic output)
    definitive_failures = [
        r"SCF not converged",
        r"not converged after \d+ cycles",
        r"Optimization Failed",
        r"Traceback \(most recent call last\)",
    ]
    for pattern in definitive_failures:
        if re.search(pattern, output, re.IGNORECASE):
            result["converged"] = False
            break

    # ── Parse dipole moment ───────────────────────────────────────────────
    # Our script prints:  Dipole moment (Debye) = [ x  y  z ]  (numpy array)
    # PySCF also prints:  Dipole moment(X, Y, Z, Debye):  x,  y,  z
    dipole_patterns = [
        # numpy array format: [ 0.  0. -1.914]
        r"Dipole moment[^=\n]*=\s*[\[(]?\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        # PySCF native: Dipole moment(X, Y, Z, Debye):  0.0,  0.0, -1.914
        r"Dipole moment\([^)]+Debye\)[^:\n]*:\s*([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)",
        # generic fallback
        r"Dipole[^:=\n]*[=:]\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    ]
    for pattern in dipole_patterns:
        dipole_match = re.search(pattern, output, re.IGNORECASE)
        if dipole_match:
            try:
                result["dipole"] = (
                    float(dipole_match.group(1)),
                    float(dipole_match.group(2)),
                    float(dipole_match.group(3)),
                )
                break
            except (ValueError, IndexError):
                pass

    # ── Parse vibrational frequencies ─────────────────────────────────────
    freq_pattern = r"Frequency\s*(?:.*?):\s*([-\d.]+)"
    freq_matches = re.findall(freq_pattern, output)
    if freq_matches:
        try:
            result["frequencies"] = [float(f) for f in freq_matches]
        except (ValueError, IndexError):
            pass

    # ── Parse thermochemistry (freq jobs) ─────────────────────────────────
    thermo_patterns = {
        "zpe": r"ZPE \(Hartree\):\s*([-\d.]+)",
        "enthalpy": r"Enthalpy \(Hartree\):\s*([-\d.]+)",
        "gibbs": r"Gibbs \(Hartree\):\s*([-\d.]+)",
        "entropy": r"Entropy \(Hartree/K\):\s*([-\d.]+)",
    }
    for key, pattern in thermo_patterns.items():
        m = re.search(pattern, output)
        if m:
            try:
                result[key] = float(m.group(1))
            except (ValueError, IndexError):
                pass

    # ── Parse frontier orbitals (HOMO/LUMO/gap) ───────────────────────────
    for key, pattern in (
        ("homo", r"HOMO \(Hartree\):\s*([-\d.]+)"),
        ("lumo", r"LUMO \(Hartree\):\s*([-\d.]+)"),
        ("homo_lumo_gap", r"HOMO-LUMO gap \(eV\):\s*([-\d.]+)"),
    ):
        m = re.search(pattern, output)
        if m:
            try:
                result[key] = float(m.group(1))
            except (ValueError, IndexError):
                pass

    # ── Parse Mulliken charges ────────────────────────────────────────────
    mulliken_section = re.search(
        r"=== MULLIKEN CHARGES ===\s*(.*?)\s*=== END MULLIKEN ===",
        output,
        re.DOTALL,
    )
    if mulliken_section:
        rows = re.findall(
            r"^\s*([A-Za-z]{1,3})\s+([-\d.]+)",
            mulliken_section.group(1),
            re.MULTILINE,
        )
        if rows:
            try:
                result["mulliken_charges"] = [(sym, float(q)) for sym, q in rows]
            except (ValueError, IndexError):
                pass

    # ── Parse optimized geometry ──────────────────────────────────────────
    # Read the explicit block the generated script prints between markers.
    geom_section = re.search(
        r"=== OPTIMIZED GEOMETRY \(Angstrom\) ===\s*(.*?)\s*=== END GEOMETRY ===",
        output,
        re.DOTALL,
    )
    if geom_section:
        atom_matches = re.findall(
            r"^\s*([A-Za-z]{1,3})\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
            geom_section.group(1),
            re.MULTILINE,
        )
        if atom_matches:
            try:
                result["geometry"] = [
                    (sym, float(x), float(y), float(z))
                    for sym, x, y, z in atom_matches
                ]
            except (ValueError, IndexError):
                pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _format_result(res: QChemResult) -> str:
    """Render a QChemResult as a compact human-readable summary for the agent."""
    lines = [
        f"Calculation finished ({res.engine}, {res.method}/{res.basis})",
        f"  Converged    : {res.converged}",
        f"  Energy       : {res.energy} Hartree" if res.energy is not None else "  Energy       : n/a",
        f"  Wall time    : {res.wall_time:.1f} s",
    ]
    if res.dipole is not None:
        lines.append(f"  Dipole (D)   : {res.dipole}")
    if res.frequencies:
        imag = [f for f in res.frequencies if f < 0]
        lines.append(f"  Frequencies  : {res.frequencies} cm^-1")
        if imag:
            lines.append(f"  ⚠ Imaginary modes: {len(imag)} (not a minimum — likely a transition state)")
    if res.zpe is not None:
        lines.append(f"  ZPE          : {res.zpe} Hartree")
    if res.enthalpy is not None:
        lines.append(f"  Enthalpy H   : {res.enthalpy} Hartree (298.15 K)")
    if res.gibbs is not None:
        lines.append(f"  Gibbs G      : {res.gibbs} Hartree (298.15 K)")
    if res.entropy is not None:
        lines.append(f"  Entropy S    : {res.entropy} Hartree/K (298.15 K)")
    if res.homo is not None:
        lines.append(f"  HOMO         : {res.homo} Hartree")
    if res.lumo is not None:
        lines.append(f"  LUMO         : {res.lumo} Hartree")
    if res.homo_lumo_gap is not None:
        lines.append(f"  HOMO-LUMO gap: {res.homo_lumo_gap} eV")
    if res.mulliken_charges:
        chg = "\n".join(f"    {s} {q:+.4f}" for s, q in res.mulliken_charges)
        lines.append(f"  Mulliken charges:\n{chg}")
    if res.geometry:
        geom = "\n".join(f"    {s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in res.geometry)
        lines.append(f"  Optimized geometry (Angstrom):\n{geom}")
    return "\n".join(lines)


def _parse_atoms(atoms: str) -> list[tuple[str, float, float, float]]:
    """Parse 'SYM x y z, SYM x y z, ...' into a list of (sym, x, y, z) tuples."""
    out: list[tuple[str, float, float, float]] = []
    for chunk in atoms.split(","):
        parts = chunk.split()
        if len(parts) >= 4:
            try:
                out.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return out


@tool
def run_calculation(
    atoms: Annotated[
        str,
        "Molecule geometry as 'SYMBOL X Y Z' lines, comma-separated "
        "(e.g. 'O 0 0 0, H 0.96 0 0, H -0.24 0.93 0'). Obtain it from parse_molecule. "
        "Required unless you pass orca_or_psi4_input instead.",
    ] = "",
    method: Annotated[str, "Method or DFT functional: HF, B3LYP, PBE0, wB97X, MP2, CCSD, ..."] = "HF",
    basis: Annotated[str, "Basis set: def2-SVP, 6-31G, STO-3G, ..."] = "def2-SVP",
    charge: Annotated[int, "Molecular charge."] = 0,
    multiplicity: Annotated[int, "Spin multiplicity (1=singlet, 2=doublet, ...)."] = 1,
    job_type: Annotated[str, "energy, opt, or freq."] = "energy",
    solvent: Annotated[str, "Solvent name, or empty for gas phase."] = "",
    orca_or_psi4_input: Annotated[
        str,
        "Alternative input: raw ORCA/Psi4 text (e.g. the ORCA block from "
        "get_molecule_from_pubchem). If given, the structured args above are ignored.",
    ] = "",
    timeout_seconds: Annotated[int, "Maximum wall time in seconds."] = 1800,
) -> str:
    """Execute a quantum chemistry calculation locally via PySCF and return the NUMBERS.

    USE THIS whenever the user asks to calculate / compute / run anything — energy,
    geometry optimization (job_type='opt'), frequencies (job_type='freq'), or
    properties (HOMO-LUMO, Mulliken charges, dipole are always reported). It actually
    runs the calculation and returns results. Do NOT use generate_config to "run" —
    that only writes an input file and does not compute anything.

    Preferred: pass the molecule as structured args (atoms/method/basis/...).
    Alternatively pass raw ORCA/Psi4 text via orca_or_psi4_input.
    """
    logger.info(
        "run_calculation: job_type={}, method={}, basis={}", job_type, method, basis
    )
    try:
        if orca_or_psi4_input.strip():
            res = run_pyscf(orca_or_psi4_input, fmt="auto", timeout_seconds=timeout_seconds)
        else:
            atom_list = _parse_atoms(atoms)
            if not atom_list:
                return (
                    "No molecule provided. Pass `atoms` as 'SYMBOL X Y Z' lines "
                    "(comma-separated) — e.g. the output of parse_molecule — or pass "
                    "orca_or_psi4_input."
                )
            job = QChemJob(
                id="run",
                method=method,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                atoms=atom_list,
                job_type=job_type if job_type in ("energy", "opt", "freq") else "energy",
                engine="pyscf",
                solvent=solvent or None,
            )
            res = _execute_pyscf_job(job, timeout_seconds=timeout_seconds)
    except (ValueError, RuntimeError) as exc:
        logger.error("run_calculation failed: {}", exc)
        return f"Calculation failed: {exc}"
    return _format_result(res)
