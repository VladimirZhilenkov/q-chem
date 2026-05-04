"""
test_run_pyscf.py
----------------
Unit and integration tests for the run_pyscf tool.

Tests:
  - Basic HF/STO-3G energy calculation
  - Input format auto-detection (ORCA vs Psi4)
  - Convergence status parsing
  - Energy extraction
  - Error handling for invalid input
"""

import pytest
from agent_tools.run_pyscf import run_pyscf, _parse_pyscf_output
from schemas import QChemResult


class TestRunPyscfBasic:
    """Test basic PySCF execution with different input formats."""

    def test_pyscf_hf_sto3g(self):
        """
        Test: HF/STO-3G single-point energy for H2O.
        Format: Psi4 input.
        Expected: Energy < 0, converged=True.
        """
        psi4_input = """
memory 4 gb

molecule {
    0 1
    O    0.000000    0.000000    0.119262
    H    0.000000    0.763239   -0.470833
    H    0.000000   -0.763239   -0.470833
}

set {
    basis sto-3g
    scf_type df
}

energy('hf')
        """

        result = run_pyscf(psi4_input, fmt="psi4", timeout_seconds=60)

        assert isinstance(result, QChemResult)
        assert result.engine == "pyscf"
        assert result.energy is not None
        assert result.energy < 0  # Hartree (atomic units)
        assert result.converged is True
        assert result.raw_output is not None
        assert len(result.raw_output) > 0

    def test_pyscf_format_autodetect(self):
        """
        Test: Auto-detection of Psi4 format.
        The converter should auto-detect 'psi4' vs 'orca'.
        """
        orca_input = """
! HF STO-3G
* xyz 0 1
O    0.0  0.0  0.119262
H    0.0  0.763239 -0.470833
H    0.0 -0.763239 -0.470833
*
        """

        result = run_pyscf(orca_input, fmt="auto", timeout_seconds=60)

        assert isinstance(result, QChemResult)
        assert result.energy is not None
        assert result.converged is True

    def test_invalid_format_raises(self):
        """Test: Invalid or undetectable input format raises ValueError."""
        invalid_input = "this is not a quantum chemistry input file"

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            run_pyscf(invalid_input, fmt="auto")

    def test_explicit_format_override(self):
        """
        Test: Explicit fmt='orca' or fmt='psi4' overrides auto-detection.
        """
        orca_input = """
! HF STO-3G
* xyz 0 1
O    0.0  0.0  0.119262
H    0.0  0.763239 -0.470833
H    0.0 -0.763239 -0.470833
*
        """

        result = run_pyscf(orca_input, fmt="orca", timeout_seconds=60)
        assert result.energy is not None


class TestParseOutput:
    """Test the output parser for PySCF results."""

    def test_parse_energy_from_output(self):
        """Test: Extract energy from PySCF output text."""
        mock_output = """
Starting PySCF calculation...
Total energy = -75.3455 Hartree
SCF converged after 8 iterations
Final result: E = -75.3455 Ha
        """
        mock_script = "mf = mol.HF(\n"

        result = _parse_pyscf_output(mock_output, mock_script, None)

        assert result["energy"] is not None
        assert result["energy"] < 0
        assert result["converged"] is True

    def test_parse_convergence_status(self):
        """Test: Extract convergence status from output."""
        mock_output = """
SCF converged in 12 iterations
Final energy: -76.4321 Hartree
        """
        mock_script = "mf = mol.KS(\n"

        result = _parse_pyscf_output(mock_output, mock_script, None)

        assert result["converged"] is True

    def test_parse_convergence_failure(self):
        """Test: Detect failed convergence."""
        mock_output = """
SCF did not converge after 200 iterations
Error: SCF not converged
        """
        mock_script = "mf = mol.HF(\n"

        result = _parse_pyscf_output(mock_output, mock_script, None)

        assert result["converged"] is False

    def test_parse_dipole_moment(self):
        """Test: Extract dipole moment from output."""
        mock_output = """
Total energy: -75.5 Ha
Dipole moment: 1.854 0.0 0.0 D
        """
        mock_script = "mf = mol.HF(\n"

        result = _parse_pyscf_output(mock_output, mock_script, None)

        assert result["dipole"] is not None
        assert len(result["dipole"]) == 3
        assert result["dipole"][0] > 0  # x-component


class TestQChemResultStructure:
    """Test that QChemResult is correctly populated."""

    def test_qchem_result_fields(self):
        """Test: All required QChemResult fields are present."""
        psi4_input = """
memory 4 gb

molecule {
    0 1
    O    0.000000    0.000000    0.119262
    H    0.000000    0.763239   -0.470833
    H    0.000000   -0.763239   -0.470833
}

set {
    basis sto-3g
}

energy('hf')
        """

        result = run_pyscf(psi4_input, fmt="psi4", timeout_seconds=60)

        # Check required fields
        assert result.job_id is not None
        assert result.energy is not None
        assert isinstance(result.converged, bool)
        assert result.wall_time > 0
        assert result.engine == "pyscf"
        assert result.method is not None
        assert result.basis is not None
        assert result.raw_output is not None
        assert len(result.raw_output) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_timeout_handling(self):
        """Test: Very short timeout should raise RuntimeError."""
        psi4_input = """
memory 4 gb

molecule {
    0 1
    He    0.0  0.0  0.0
}

set basis sto-3g
energy('hf')
        """

        # Set timeout to 0.001 seconds — almost certainly will timeout
        with pytest.raises(RuntimeError, match="timed out"):
            run_pyscf(psi4_input, fmt="psi4", timeout_seconds=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
