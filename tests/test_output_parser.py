"""Unit tests for PySCF output parsing (regex layer, no execution)."""

import pytest

from agent.agent_tools.run_pyscf import _parse_pyscf_output

# Minimal generated-script stub used only for method/basis extraction.
SCRIPT = "method = 'b3lyp'\nmol = gto.M(basis='sto-3g')\n"


def _parse(output, tmp_path):
    return _parse_pyscf_output(output, SCRIPT, tmp_path)


def test_energy_priority_during_opt(tmp_path):
    # During an opt PySCF prints one "converged SCF energy" per geometry step;
    # the authoritative value is the final "Energy (Hartree):" line, not the first SCF.
    output = (
        "converged SCF energy = -74.95985317\n"
        "converged SCF energy = -74.96400000\n"
        "Energy (Hartree): -74.9659011867\n"
        "Converged: True\n"
    )
    assert _parse(output, tmp_path)["energy"] == pytest.approx(-74.9659011867)


def test_frequencies_preserve_imaginary(tmp_path):
    output = (
        "=== VIBRATIONAL FREQUENCIES (cm^-1) ===\n"
        "Frequency 1: -1200.5000\n"
        "Frequency 2: 1500.0000\n"
        "Energy (Hartree): -74.0\n"
    )
    assert _parse(output, tmp_path)["frequencies"] == [-1200.5, 1500.0]


def test_thermochemistry(tmp_path):
    output = (
        "=== THERMOCHEMISTRY (298.15 K) ===\n"
        "ZPE (Hartree): 0.02582536\n"
        "Enthalpy (Hartree): -74.93333875\n"
        "Gibbs (Hartree): -74.95540110\n"
        "Entropy (Hartree/K): 0.0000739975\n"
    )
    r = _parse(output, tmp_path)
    assert r["zpe"] == pytest.approx(0.02582536)
    assert r["enthalpy"] == pytest.approx(-74.93333875)
    assert r["gibbs"] == pytest.approx(-74.95540110)
    assert r["entropy"] == pytest.approx(0.0000739975)


def test_frontier_orbitals(tmp_path):
    output = (
        "HOMO (Hartree): -0.144228\n"
        "LUMO (Hartree): 0.355975\n"
        "HOMO-LUMO gap (eV): 13.6112\n"
    )
    r = _parse(output, tmp_path)
    assert r["homo"] == pytest.approx(-0.144228)
    assert r["lumo"] == pytest.approx(0.355975)
    assert r["homo_lumo_gap"] == pytest.approx(13.6112)


def test_mulliken_charges(tmp_path):
    output = (
        "=== MULLIKEN CHARGES ===\n"
        "O -0.368700\n"
        "H 0.184400\n"
        "H 0.184300\n"
        "=== END MULLIKEN ===\n"
    )
    assert _parse(output, tmp_path)["mulliken_charges"] == [
        ("O", pytest.approx(-0.3687)),
        ("H", pytest.approx(0.1844)),
        ("H", pytest.approx(0.1843)),
    ]


def test_optimized_geometry(tmp_path):
    output = (
        "=== OPTIMIZED GEOMETRY (Angstrom) ===\n"
        "O -0.031115 -0.043966 0.000000\n"
        "H 0.955646 0.028892 -0.000000\n"
        "=== END GEOMETRY ===\n"
    )
    geom = _parse(output, tmp_path)["geometry"]
    assert geom[0][0] == "O"
    assert geom[1][1] == pytest.approx(0.955646)


def test_method_label_from_runtime_variable(tmp_path):
    # The whole if/elif dispatch is present in the script as text; the label must
    # come from the runtime `method = '...'` line, not from `scf.RHF(`.
    script = "method = 'b3lyp'\nif x:\n    mf = scf.RHF(mol)\n"
    assert _parse_pyscf_output("Energy (Hartree): -1.0\n", script, tmp_path)["method"] == "b3lyp"
