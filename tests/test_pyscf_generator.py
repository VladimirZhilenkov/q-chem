"""Unit tests for the PySCF script generator (text only, no execution)."""

from config_generators import generate_pyscf
from schemas import QChemJob


def _job(method="b3lyp", basis="def2-SVP", mult=1, job_type="energy"):
    return QChemJob(
        id="t",
        method=method,
        basis=basis,
        charge=0,
        multiplicity=mult,
        atoms=[("O", 0.0, 0.0, 0.0), ("H", 0.96, 0.0, 0.0), ("H", -0.24, 0.93, 0.0)],
        job_type=job_type,
    )


def test_method_and_basis_templated():
    script = generate_pyscf(_job(method="PBE0", basis="def2-SVP"))
    assert "method = 'pbe0'" in script
    assert "basis='def2-SVP'" in script


def test_open_shell_sets_spin():
    # multiplicity 2 → spin (2S) = 1
    assert "spin=1" in generate_pyscf(_job(mult=2))
    assert "spin=0" in generate_pyscf(_job(mult=1))


def test_opt_branch_prints_geometry():
    script = generate_pyscf(_job(job_type="opt"))
    assert "geometric_solver import optimize" in script
    assert "OPTIMIZED GEOMETRY" in script


def test_freq_branch_has_thermo():
    script = generate_pyscf(_job(job_type="freq"))
    assert "harmonic_analysis" in script
    assert "THERMOCHEMISTRY" in script


def test_dispersion_method_preserved_with_guard():
    script = generate_pyscf(_job(method="B3LYP-D3BJ"))
    assert "method = 'b3lyp-d3bj'" in script
    assert "mf.disp" in script            # dispersion is wired
    assert "pyscf.dispersion" in script   # backend-availability guard present


def test_properties_always_printed():
    script = generate_pyscf(_job())
    assert "HOMO-LUMO gap" in script
    assert "MULLIKEN CHARGES" in script
