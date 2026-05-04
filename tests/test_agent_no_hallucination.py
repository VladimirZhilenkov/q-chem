"""
End-to-end no-hallucination tests for the agent.

For each molecule in `tests/data/molecules.py`, three layers are checked:
  1. converter+run_pyscf on the ORCA input  → reference energy
  2. converter+run_pyscf on the Psi4 input  → reference energy
  3. full agent (`run_agent`) on a natural-language prompt with the input
     embedded                               → reference energy parsed from
                                              the agent's final answer

If any layer's energy disagrees with the reference by more than the
tolerance, the test fails. The agent layer (3) is the strict
no-hallucination check: only a real call to `run_pyscf` would produce a
matching number.

Run:
    pytest tests/test_agent_no_hallucination.py -v

Skip the agent-layer tests (slow + requires LLM access):
    pytest tests/test_agent_no_hallucination.py -v -m "not agent"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent.agent_tools.run_pyscf import run_pyscf
from tests.data.input_generators import make_orca_input, make_psi4_input
from tests.data.molecules import DATASET


REFERENCES_PATH = Path(__file__).parent / "data" / "references.json"
TOLERANCE_HA = 1e-3  # 1 milli-Hartree


def _key(mol) -> str:
    return f"{mol['name']}__{mol['method']}__{mol['basis']}__{mol['job_type']}"


def _load_references() -> dict:
    if not REFERENCES_PATH.exists():
        pytest.skip(
            f"{REFERENCES_PATH} not found. "
            "Run `python -m tests.generate_references` first."
        )
    return json.loads(REFERENCES_PATH.read_text())


REFERENCES = _load_references() if REFERENCES_PATH.exists() else {}


def _expected_energy(mol) -> float:
    ref = REFERENCES.get(_key(mol))
    if ref is None or "energy_ha" not in ref:
        pytest.skip(f"No reference for {_key(mol)}")
    return ref["energy_ha"]


@pytest.mark.parametrize("mol", DATASET, ids=lambda m: _key(m))
def test_orca_input_matches_reference(mol):
    """run_pyscf on ORCA input must reproduce the local PySCF reference."""
    expected = _expected_energy(mol)
    result = run_pyscf(make_orca_input(mol), fmt="orca")
    assert result.energy is not None, f"No energy parsed for {_key(mol)}"
    assert abs(result.energy - expected) < TOLERANCE_HA, (
        f"{_key(mol)}: ORCA path got {result.energy:.8f} Ha, "
        f"ref {expected:.8f} Ha, "
        f"|Δ|={abs(result.energy-expected)*1000:.3f} mHa > {TOLERANCE_HA*1000} mHa"
    )


@pytest.mark.parametrize("mol", DATASET, ids=lambda m: _key(m))
def test_psi4_input_matches_reference(mol):
    """run_pyscf on Psi4 input must reproduce the local PySCF reference."""
    expected = _expected_energy(mol)
    result = run_pyscf(make_psi4_input(mol), fmt="psi4")
    assert result.energy is not None, f"No energy parsed for {_key(mol)}"
    assert abs(result.energy - expected) < TOLERANCE_HA, (
        f"{_key(mol)}: Psi4 path got {result.energy:.8f} Ha, "
        f"ref {expected:.8f} Ha, "
        f"|Δ|={abs(result.energy-expected)*1000:.3f} mHa > {TOLERANCE_HA*1000} mHa"
    )


_ENERGY_RE = re.compile(r"(-?\d+\.\d{4,})\s*(?:Ha|Hartree|hartree)?", re.IGNORECASE)


def _extract_energy_from_text(text: str) -> float | None:
    """Pull the first plausible Hartree-magnitude energy from agent output."""
    candidates = []
    for m in _ENERGY_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if -1000.0 < v < 0.0:  # Plausible total-energy range in Hartree
            candidates.append(v)
    return candidates[0] if candidates else None


@pytest.mark.agent
@pytest.mark.parametrize("fmt", ["orca", "psi4"])
@pytest.mark.parametrize("mol", DATASET, ids=lambda m: _key(m))
def test_agent_matches_reference(mol, fmt):
    """Full agent must call run_pyscf for real and return a matching energy.

    If the LLM hallucinates, the energy in its final answer either won't
    parse or won't match — both are failures.
    """
    from agent.graph import run_agent

    expected = _expected_energy(mol)
    raw_input = make_orca_input(mol) if fmt == "orca" else make_psi4_input(mol)
    prompt = (
        f"Посчитай следующую задачу и верни total energy в Hartree:\n\n{raw_input}"
    )

    answer = run_agent(prompt)
    energy = _extract_energy_from_text(answer)
    assert energy is not None, (
        f"{_key(mol)} ({fmt}): no Hartree-like number found in agent answer:\n{answer}"
    )
    assert abs(energy - expected) < TOLERANCE_HA, (
        f"{_key(mol)} ({fmt}): agent returned {energy:.8f} Ha, "
        f"ref {expected:.8f} Ha, "
        f"|Δ|={abs(energy-expected)*1000:.3f} mHa > {TOLERANCE_HA*1000} mHa.\n"
        f"Full agent answer:\n{answer}"
    )
