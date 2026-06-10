"""Tool: парсинг ORCA/Psi4 входа в текстовое резюме."""

from typing import Annotated

from langchain_core.tools import tool

from converter.qchem_converter import OrcaParser, Psi4Parser


def _detect_format(text: str) -> str:
    if "!" in text or text.strip().startswith("!"):
        return "orca"
    if "molecule" in text:
        return "psi4"
    raise ValueError("Cannot detect format (expected ORCA '!' or Psi4 'molecule').")


@tool
def standardize_chem_input(
    raw_format: Annotated[str, "Raw ORCA or Psi4 input text."],
    input_format: Annotated[str, "Format: auto (default), orca, or psi4."] = "auto",
) -> str:
    """Parse ORCA or Psi4 input and return a normalized text summary."""
    fmt = input_format if input_format != "auto" else _detect_format(raw_format)

    if fmt == "orca":
        job = OrcaParser().parse(raw_format)
    elif fmt == "psi4":
        job = Psi4Parser().parse(raw_format)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    atoms = ", ".join(atom[0] for atom in job.atoms[:5])
    if len(job.atoms) > 5:
        atoms += f", ... (+{len(job.atoms) - 5})"

    return (
        f"Molecule: {len(job.atoms)} atoms ({atoms})\n"
        f"Method: {job.method}\n"
        f"Basis: {job.basis}\n"
        f"Charge: {job.charge}, Multiplicity: {job.multiplicity}\n"
        f"Job type: {job.job_type}"
    )
