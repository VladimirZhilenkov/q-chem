"""Tool: запуск расчёта через PySCF."""

import json
from typing import Annotated

from langchain_core.tools import tool

from .run_pyscf import run_pyscf as _run_pyscf_impl


@tool
def run_pyscf(
    orca_or_psi4_input: Annotated[str, "Raw ORCA or Psi4 input text."],
) -> str:
    """Run a quantum chemistry calculation via PySCF."""
    raw = orca_or_psi4_input.strip()

    # LLM иногда оборачивает вход в JSON — распаковываем
    if raw.startswith("{") and raw.endswith("}"):
        try:
            data = json.loads(raw)
            for key in ("orca_or_psi4_input", "raw_format", "input"):
                if isinstance(data.get(key), str):
                    raw = data[key]
                    break
        except json.JSONDecodeError:
            pass

    if "\\n" in raw:
        raw = raw.replace("\\n", "\n").replace("\\t", "\t")

    result = _run_pyscf_impl(raw, fmt="auto")

    energy = f"{result.energy:.8f} Hartree" if result.energy is not None else "N/A"
    dipole = "N/A"
    if result.dipole is not None:
        x, y, z = result.dipole
        dipole = f"({x:.4f}, {y:.4f}, {z:.4f}) Debye"

    return (
        f"Energy: {energy}\n"
        f"Converged: {result.converged}\n"
        f"Method: {result.method}\n"
        f"Basis: {result.basis}\n"
        f"Dipole: {dipole}\n"
        f"Wall time: {result.wall_time:.2f} s"
    )
