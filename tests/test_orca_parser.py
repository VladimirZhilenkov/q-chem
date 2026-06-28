"""Unit tests for the ORCA → QChemJob parser.

These lock in the route-line parsing fixes: run-type keywords (SP/Opt/Freq) and
auxiliary modifiers (RIJCOSX, D3BJ, TightSCF) must not be mistaken for the method.
Fast and deterministic — no PySCF, no network.
"""

import pytest

from converter.qchem_converter import parse


def _orca(route: str, charge: int = 0, mult: int = 1) -> str:
    return f"! {route}\n* xyz {charge} {mult}\n  O 0 0 0\n  H 0.96 0 0\n  H -0.24 0.93 0\n*\n"


def test_sp_keyword_is_not_the_method():
    # Regression: "! HF STO-3G SP" used to parse method='sp'
    job = parse(_orca("HF STO-3G SP"), fmt="orca")
    assert job.method.lower() == "hf"
    assert job.basis.lower() == "sto-3g"
    assert job.job_type == "energy"


def test_modifiers_are_skipped():
    job = parse(_orca("RIJCOSX B3LYP def2-TZVP D3BJ TightSCF SP"), fmt="orca")
    assert job.method.lower() == "b3lyp"
    assert job.basis.lower() == "def2-tzvp"


def test_real_hf_reference_preserved():
    # UHF/RHF/ROHF are genuine methods and must survive the keyword filter
    job = parse(_orca("UHF 6-31G SP", charge=0, mult=2), fmt="orca")
    assert job.method.lower() == "uhf"
    assert job.multiplicity == 2


@pytest.mark.parametrize(
    "route,expected",
    [("B3LYP def2-SVP Opt", "opt"), ("HF STO-3G Freq", "freq"), ("PBE0 def2-SVP", "energy")],
)
def test_job_type_detection(route, expected):
    assert parse(_orca(route), fmt="orca").job_type == expected


def test_coordinates_and_charge():
    job = parse(_orca("HF STO-3G SP", charge=-1, mult=2), fmt="orca")
    assert job.charge == -1
    assert len(job.atoms) == 3
    assert job.atoms[0][0] == "O"
