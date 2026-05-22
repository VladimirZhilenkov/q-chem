import json

from langchain_core.tools import tool

from .run_pyscf import run_pyscf as _run_pyscf_impl


@tool
def run_pyscf(orca_or_psi4_input: str) -> str:
    """Run a quantum chemistry calculation. The input MUST be the original
    ORCA input (containing a `!` keyword line and `* xyz ... *` block) or
    the original Psi4 input (containing a `molecule { ... }` block) as a
    single plain text string. Do NOT wrap the input in JSON or any other
    structure — pass the raw input text directly. The format is
    auto-detected. Returns a short summary with total energy in Hartree,
    convergence flag, method, and basis."""
    raw = orca_or_psi4_input

    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                for k in ("orca_or_psi4_input", "raw_format", "input"):
                    if k in parsed and isinstance(parsed[k], str):
                        raw = parsed[k]
                        break
        except json.JSONDecodeError:
            pass

    # Normalize escaped newlines regardless of whether real newlines also exist.
    # The LLM sometimes double-encodes \n (e.g. \\n) even after JSON parsing.
    if "\\n" in raw:
        raw = raw.replace("\\n", "\n").replace("\\t", "\t")

    result = _run_pyscf_impl(raw, fmt="auto")
    energy_str = (
        f"{result.energy:.8f} Hartree" if result.energy is not None else "N/A"
    )
    dipole_str = (
        f"({result.dipole[0]:.4f}, {result.dipole[1]:.4f}, {result.dipole[2]:.4f}) Debye"
        if result.dipole is not None
        else "N/A"
    )
    return (
        f"Energy: {energy_str}\n"
        f"Converged: {result.converged}\n"
        f"Method: {result.method}\n"
        f"Basis: {result.basis}\n"
        f"Dipole: {dipole_str}\n"
        f"Wall time: {result.wall_time:.2f} s"
    )


__all__ = ["run_pyscf"]
