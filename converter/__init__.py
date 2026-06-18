"""Converter module - parse quantum chemistry input files.

Functionality:
  • Parse ORCA and Psi4 input files into QChemJob specs
  • Auto-detect format
  • Validate basis sets
"""

from converter.qchem_converter import Psi4Parser, OrcaParser, parse, detect_format

__all__ = ["Psi4Parser", "OrcaParser", "parse", "detect_format"]
