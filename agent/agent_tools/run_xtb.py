"""
run_xtb.py
----------
Execute a QChemJob with the xTB semiempirical tight-binding program.

xTB is a separate binary (not a Python library like PySCF), so this shells out
to the `xtb` executable, runs in a temp directory, and parses stdout plus the
files xTB writes (xtbopt.xyz, vibspectrum) into a QChemResult.
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from loguru import logger

from schemas import QChemJob, QChemResult

# Allow overriding the xtb executable (e.g. a conda/WSL path) via .env.
_XTB_BIN = os.getenv("XTB_BIN", "xtb")

# Map the method names the agent uses onto xTB's `--gfn` Hamiltonian level.
# ('gfnff' is selected with a dedicated flag rather than --gfn.)
_GFN_LEVEL = {
    "gfn0": "0",
    "gfn0-xtb": "0",
    "gfn1": "1",
    "gfn1-xtb": "1",
    "gfn2": "2",
    "gfn2-xtb": "2",
}


def _write_xyz(job: QChemJob, path: Path) -> None:
    """Write the job geometry as a standard .xyz file."""
    lines = [str(len(job.atoms)), f"xTB calculation: {job.id}"]
    for symbol, x, y, z in job.atoms:
        lines.append(f"{symbol:2} {x:14.8f} {y:14.8f} {z:14.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_command(job: QChemJob, xyz_name: str) -> list[str]:
    """Assemble the xtb command line from the job specification."""
    method = job.method.lower()
    cmd = [_XTB_BIN, xyz_name]

    if method in ("gfnff", "gfn-ff", "gff"):
        cmd.append("--gfnff")
    else:
        cmd += ["--gfn", _GFN_LEVEL.get(method, "2")]

    cmd += ["--chrg", str(job.charge)]
    # xTB wants the number of UNPAIRED electrons, not the spin multiplicity.
    cmd += ["--uhf", str(max(job.multiplicity - 1, 0))]

    # Job type: --opt (geometry), --ohess (opt + frequencies), --hess (freq only).
    if job.job_type == "opt":
        cmd.append("--opt")
    elif job.job_type == "freq":
        cmd.append("--hess")

    # Implicit solvent via the ALPB model (accepts a named solvent like 'water').
    if job.solvent:
        cmd += ["--alpb", job.solvent]

    return cmd


def _parse_xtb_output(output: str, tmpdir: Path) -> dict:
    """Parse xtb stdout plus written files into the result dict."""
    result: dict = {
        "energy": None,
        "converged": False,
        "geometry": None,
        "frequencies": None,
        "homo": None,
        "lumo": None,
        "homo_lumo_gap": None,
        "dipole": None,
    }

    # Total energy (Hartree). xtb prints e.g. "| TOTAL ENERGY  -5.0848 Eh |".
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh", output, re.IGNORECASE)
    if m:
        result["energy"] = float(m.group(1))

    # HOMO-LUMO gap in eV.
    m = re.search(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)\s*eV", output, re.IGNORECASE)
    if m:
        result["homo_lumo_gap"] = float(m.group(1))

    result["converged"] = "normal termination of xtb" in output

    # Optimised geometry: xtb writes the final structure to xtbopt.xyz.
    opt_xyz = tmpdir / "xtbopt.xyz"
    if opt_xyz.exists():
        geom = []
        for line in opt_xyz.read_text(encoding="utf-8").splitlines()[2:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    geom.append(
                        (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                    )
                except ValueError:
                    continue
        if geom:
            result["geometry"] = geom

    # Vibrational frequencies: xtb writes a Turbomole-style `vibspectrum` file.
    vib = tmpdir / "vibspectrum"
    if vib.exists():
        freqs = []
        for line in vib.read_text(encoding="utf-8").splitlines():
            # Data rows look like:  1   2  1234.56  ... ; column 3 is cm^-1.
            mm = re.match(r"\s*\d+\s+\d+\s+(-?\d+\.\d+)", line)
            if mm:
                freqs.append(float(mm.group(1)))
        # Drop the zero (translation/rotation) modes xtb lists as 0.0.
        freqs = [f for f in freqs if abs(f) > 1e-3]
        if freqs:
            result["frequencies"] = freqs

    return result


def execute_xtb_job(job: QChemJob, timeout_seconds: int = 1800) -> QChemResult:
    """Run a QChemJob with xTB and return a parsed QChemResult."""
    if shutil.which(_XTB_BIN) is None:
        raise RuntimeError(
            f"The xTB executable '{_XTB_BIN}' was not found on PATH. Install it "
            "(e.g. `conda install -c conda-forge xtb`) or set XTB_BIN in .env to "
            "its full path."
        )

    start_time = time.time()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        xyz_name = "struct.xyz"
        _write_xyz(job, tmpdir_path / xyz_name)
        cmd = _build_command(job, xyz_name)
        logger.info("run_xtb: {}", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmpdir_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            wall_time = time.time() - start_time
            raise RuntimeError(
                f"xTB calculation timed out after {timeout_seconds}s "
                f"(elapsed {wall_time:.1f}s)."
            ) from e

        raw_output = proc.stdout + "\n" + proc.stderr
        wall_time = time.time() - start_time

        if proc.returncode != 0:
            raise RuntimeError(
                f"xTB failed with return code {proc.returncode}. "
                f"Output:\n{raw_output.strip()}"
            )

        parsed = _parse_xtb_output(raw_output, tmpdir_path)

    return QChemResult(
        job_id=job.id,
        energy=parsed["energy"],
        converged=parsed["converged"],
        geometry=parsed["geometry"],
        frequencies=parsed["frequencies"],
        dipole=parsed["dipole"],
        homo=parsed["homo"],
        lumo=parsed["lumo"],
        homo_lumo_gap=parsed["homo_lumo_gap"],
        mulliken_charges=None,
        wall_time=wall_time,
        engine="xtb",
        method=job.method.lower(),
        basis="(semiempirical)",
        raw_output=raw_output,
        output_path=None,
    )
