"""Integration tests: actually execute PySCF via the run_calculation tool.

Small systems (H2 / water at STO-3G) run in well under a second. Marked `slow`
so they can be skipped with `-m "not slow"`.
"""

import pytest

from agent.agent_tools.run_pyscf import run_calculation

pytestmark = pytest.mark.slow


def _run(route, charge=0, mult=1, atoms="  H 0 0 0\n  H 0 0 0.74"):
    inp = f"! {route}\n* xyz {charge} {mult}\n{atoms}\n*\n"
    return run_calculation.invoke({"orca_or_psi4_input": inp})


def test_energy_hf_h2():
    out = _run("HF STO-3G SP")
    assert "Converged    : True" in out
    assert "-1.116" in out  # HF/STO-3G H2 ≈ -1.1168 Hartree


def test_direct_molecule_path():
    # Compute straight from a molecule spec — no ORCA round-trip
    out = run_calculation.invoke(
        {"atoms": "O 0 0 0, H 0.957 0 0, H -0.239 0.927 0", "method": "HF", "basis": "STO-3G"}
    )
    assert "Converged    : True" in out
    assert "-74.96" in out  # HF/STO-3G water ≈ -74.96 Hartree


def test_missing_molecule_is_handled():
    out = run_calculation.invoke({"atoms": ""})
    assert "No molecule provided" in out


def test_dft_functional_runs():
    # A functional that is NOT b3lyp — exercises the generalized DFT dispatch
    out = _run("PBE0 STO-3G SP")
    assert "pbe0/sto-3g" in out
    assert "Energy" in out


def test_open_shell_uhf():
    out = _run("UHF STO-3G SP", mult=2, atoms="  H 0 0 0")
    assert "Converged    : True" in out


def test_optimization_lowers_energy():
    out = _run("HF STO-3G Opt", atoms="  H 0 0 0\n  H 0 0 0.90")
    assert "Optimized geometry" in out
    assert "Converged    : True" in out


def test_frequencies_and_thermo():
    out = _run("HF STO-3G Freq")
    assert "Frequencies" in out
    assert "Gibbs G" in out
    assert "ZPE" in out


def test_molecular_properties():
    out = _run("B3LYP STO-3G SP", atoms="  O 0 0 0\n  H 0.957 0 0\n  H -0.239 0.927 0")
    assert "HOMO-LUMO gap" in out
    assert "Mulliken charges" in out


def test_dispersion_without_backend_errors_cleanly():
    # No dftd3/dftd4 backend installed → actionable error, not a raw traceback
    out = _run("B3LYP-D3BJ STO-3G SP")
    assert "Calculation failed" in out
    assert "dftd3/dftd4" in out
