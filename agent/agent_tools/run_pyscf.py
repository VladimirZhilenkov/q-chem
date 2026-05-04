"""
run_pyscf.py
-----------
LangChain tool: convert ORCA/Psi4 input → PySCF script → execute → parse result.

Takes raw ORCA or Psi4 input as string, converts to PySCF, runs it via subprocess,
and returns QChemResult with energy, converged status, geometry, etc.
"""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from schemas import QChemResult
from converter.qchem_converter import convert, detect_format


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
    start_time = time.time()

    # Step 1: Auto-detect format if needed
    if fmt == "auto":
        fmt = detect_format(orca_or_psi4_input, "input")
        if fmt == "unknown":
            raise ValueError(
                "Cannot auto-detect input format. "
                "Pass fmt='psi4' or fmt='orca' explicitly."
            )

    # Step 2: Convert to PySCF script
    pyscf_script = convert(
        orca_or_psi4_input, fmt=fmt, source_name="input"
    )

    # Step 3: Execute the PySCF script in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "calc.py"
        output_path = tmpdir_path / "output.txt"

        # Write the generated script with UTF-8 encoding
        script_path.write_text(pyscf_script, encoding="utf-8")

        # Run the script via subprocess, capturing stdout and stderr
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
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
            wall_time=wall_time,
            engine="pyscf",
            method=parsed.get("method", "unknown"),
            basis=parsed.get("basis", "unknown"),
            raw_output=raw_output,
            output_path=str(output_path) if output_path.exists() else None,
        )

        return qchem_result


def _parse_pyscf_output(
    output: str, pyscf_script: str, tmpdir: Path
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
        "method": "unknown",
        "basis": "unknown",
    }

    # Extract method and basis from the PySCF script
    method_match = re.search(r"mf\s*=\s*mols\.(\w+|pyscf\.\w+\.(\w+))", pyscf_script)
    basis_match = re.search(r"\.build\(\s*['\"](\w+-?\w*)['\"]", pyscf_script)
    
    if method_match:
        result["method"] = method_match.group(1).lower()
    if basis_match:
        result["basis"] = basis_match.group(1).lower()

    # Parse energy: look for "Total energy" or "E =" patterns
    energy_patterns = [
        r"Total energy\s*=\s*([-\d.]+)",  # PySCF standard output
        r"E\s*=\s*([-\d.]+)\s*Hartree",
        r"Final energy:\s*([-\d.]+)",
        r"@.*E\(.*\)\s*=\s*([-\d.]+)",  # CC output
    ]
    for pattern in energy_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                result["energy"] = float(match.group(1))
                break
            except (ValueError, IndexError):
                pass

    # Parse convergence status
    converged_patterns = [
        r"converged\s*=\s*True",
        r"SCF converged",
        r"Optimization successful",
        r"Converged in \d+ iterations",
    ]
    for pattern in converged_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            result["converged"] = True
            break

    # Check for failure/error indicators
    error_patterns = [
        r"Error",
        r"FAILED",
        r"Exception",
        r"not converged",
        r"Optimization Failed",
    ]
    for pattern in error_patterns:
        if re.search(pattern, output):
            result["converged"] = False
            break

    # Parse dipole moment: extract (x, y, z) components
    dipole_pattern = r"Dipole[^:]*:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    dipole_match = re.search(dipole_pattern, output)
    if dipole_match:
        try:
            result["dipole"] = (
                float(dipole_match.group(1)),
                float(dipole_match.group(2)),
                float(dipole_match.group(3)),
            )
        except (ValueError, IndexError):
            pass

    # Parse frequencies (vibrational)
    freq_pattern = r"Frequency .*:\s*([-\d.]+)"
    freq_matches = re.findall(freq_pattern, output)
    if freq_matches:
        try:
            result["frequencies"] = [float(f) for f in freq_matches]
        except (ValueError, IndexError):
            pass

    # Parse final geometry if geometry optimization was performed
    geom_section = re.search(
        r"Final structure|Final coordinates|Geometry optimization complete:?\s*" +
        r"((?:\s+\w+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+)+)",
        output,
        re.IGNORECASE | re.DOTALL
    )
    if geom_section:
        geom_text = geom_section.group(1) if len(geom_section.groups()) > 0 else output
        atom_pattern = r"^\s*(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        atom_matches = re.findall(atom_pattern, geom_text, re.MULTILINE)
        if atom_matches:
            try:
                result["geometry"] = [
                    (symbol, float(x), float(y), float(z))
                    for symbol, x, y, z in atom_matches
                ]
            except (ValueError, IndexError):
                pass

    return result
