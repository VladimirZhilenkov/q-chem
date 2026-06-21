"""
qchem_converter.py
------------------
Parse Psi4 and ORCA input files into QChemJob specifications.

Converts between different input formats and normalizes to QChemJob.
Does NOT generate configs – that's handled by config_generators/ module.

Usage:
    from converter.qchem_converter import Psi4Parser, OrcaParser, detect_format
    
    parser = Psi4Parser()
    job = parser.parse(psi4_text)
"""

import re
from pathlib import Path
from typing import Optional
from schemas import QChemJob
from converter.basis_utils import check_existence

def basis(raw: str) -> str:
    """Define basis name and check existence via BSE."""
    if check_existence(raw):
        return raw
    raise ValueError(f"Basis set '{raw}' not found in Basis Set Exchange")


# ─────────────────────────────────────────────────────────────────────────────
# Psi4 parser
# ─────────────────────────────────────────────────────────────────────────────

class Psi4Parser:
    """Parse a Psi4 input file (.inp or .dat) into a QChemJob."""

    def parse(self, text: str, job_id: str = "psi4_job") -> QChemJob:
        """Parse Psi4 input and return QChemJob instance."""
        text = self._strip_comments(text)
        
        # Parse components
        atoms, charge, multiplicity = self._parse_molecule(text)
        method = self._parse_method(text)
        basis = self._parse_basis(text)
        job_type = self._parse_job_type(text)
        solvent = self._parse_solvent(text)
        nprocs = self._parse_nprocs(text)
        memory_mb = self._parse_memory(text)
        
        # Create and return QChemJob
        return QChemJob(
            id=job_id,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            atoms=atoms,
            job_type=job_type,
            engine="pyscf",
            solvent=solvent,
            nprocs=nprocs,
            memory_mb=memory_mb
        )

    # ------------------------------------------------------------------
    def _strip_comments(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            # Psi4 comments start with #
            line = re.sub(r'#.*$', '', line)
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _parse_molecule(self, text: str) -> tuple[list[tuple[str, float, float, float]], int, int]:
        """Extract atoms, charge, multiplicity from molecule{} block."""
        atoms = []
        charge, mult = 0, 1

        mol_pattern = re.compile(
            r'molecule\s*\w*\s*\{(.*?)\}',
            re.DOTALL | re.IGNORECASE
        )
        m = mol_pattern.search(text)
        if not m:
            return atoms, charge, mult

        block = m.group(1).strip()
        lines = [l.strip() for l in block.splitlines() if l.strip()]

        # First non-blank line is charge / multiplicity
        if lines:
            first = lines[0].split()
            if len(first) >= 2 and all(t.lstrip('-').isdigit() for t in first[:2]):
                charge = int(first[0])
                mult = int(first[1])
                lines = lines[1:]

        # Remaining lines are coordinates
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                sym = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atoms.append((sym, x, y, z))
                except ValueError:
                    pass  # might be a Z-matrix line – skip for now
            elif len(parts) == 1 and parts[0].lower() in ('--', 'units', 'symmetry'):
                pass  # fragment separator or option – skip

        return atoms, charge, mult

    # ------------------------------------------------------------------
    def _parse_basis(self, text: str) -> str:
        """Extract basis set from set block or inline."""
        basis_name = "sto-3g"  # default
        
        # Check set block
        set_pattern = re.compile(r'set\s*\{(.*?)\}', re.DOTALL | re.IGNORECASE)
        for m in set_pattern.finditer(text):
            block = m.group(1)
            for line in block.splitlines():
                if 'basis' in line.lower():
                    parts = re.split(r'[\s=]+', line.strip(), maxsplit=1)
                    if len(parts) == 2:
                        basis_name = basis(parts[1].strip().strip("'\""))

        # Check inline set basis
        inline = re.compile(r'set\s+basis\s*=?\s*(\S+)', re.IGNORECASE)
        m = inline.search(text)
        if m:
            basis_name = basis(m.group(1).strip().strip("'\""))

        return basis_name

    # ------------------------------------------------------------------
    def _parse_method(self, text: str) -> str:
        """Extract method from energy/optimize/frequencies call."""
        method = "hf"  # default
        
        # Look for energy('method'), optimize('method'), frequencies('method')
        task_pattern = re.compile(
            r'(energy|optimize|frequencies)\s*\(\s*[\'"]?(\w+(?:\([tT]\))?)[\'"]?\s*\)',
            re.IGNORECASE
        )
        m = task_pattern.search(text)
        if m:
            method = m.group(2).strip()

        return method

    # ------------------------------------------------------------------
    def _parse_job_type(self, text: str) -> str:
        """Determine job type from task calls."""
        if re.search(r'frequencies\s*\(', text, re.IGNORECASE):
            return "freq"
        elif re.search(r'optimize\s*\(', text, re.IGNORECASE):
            return "opt"
        else:
            return "energy"

    # ------------------------------------------------------------------
    def _parse_solvent(self, text: str) -> Optional[str]:
        """Extract solvent model if present."""
        # Look for pcm or ddcosmo in set block
        if re.search(r'pcm\s*=\s*true', text, re.IGNORECASE):
            return "pcm"
        if re.search(r'ddcosmo', text, re.IGNORECASE):
            return "ddcosmo"
        return None

    # ------------------------------------------------------------------
    def _parse_memory(self, text: str) -> int:
        """Extract memory setting in MB."""
        # Look for memory or set_memory
        mem_pattern = re.compile(r'memory\s+(\d+(?:\.\d+)?)\s*(\w+)', re.IGNORECASE)
        m = mem_pattern.search(text)
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ('gb', 'g'):
                return int(value * 1024)
            elif unit in ('mb', 'm'):
                return int(value)
        return 4000  # default

    # ------------------------------------------------------------------
    def _parse_nprocs(self, text: str) -> int:
        """Extract number of processors."""
        # Look for set_num_threads
        nproc_pattern = re.compile(r'set_num_threads\s*\(\s*(\d+)\s*\)', re.IGNORECASE)
        m = nproc_pattern.search(text)
        if m:
            return int(m.group(1))
        return 1  # default


# ─────────────────────────────────────────────────────────────────────────────
# ORCA parser
# ─────────────────────────────────────────────────────────────────────────────

# ORCA "! ..." route-line tokens that are NOT methods: run types and common
# accuracy / auxiliary / dispersion modifiers. Used to avoid mistaking them for
# the method when parsing the simple-input line.
_ORCA_NON_METHOD_KEYWORDS = {
    # run types
    "sp", "energy", "opt", "copt", "zopt", "freq", "numfreq", "anfreq",
    "optfreq", "engrad", "numgrad", "md", "scants", "neb", "neb-ts", "irc",
    # SCF convergence / accuracy
    "tightscf", "verytightscf", "normalscf", "loosescf", "sloppyscf",
    "extremescf", "slowconv", "veryslowconv", "kdiis", "soscf", "diis", "noiter",
    # grids / RI / aux
    "rijcosx", "rijk", "ri", "nori", "ri-jk", "ri-j", "autoaux", "noautostart",
    "defgrid1", "defgrid2", "defgrid3", "grid4", "grid5", "grid6", "finalgrid6",
    "nofinalgrid", "gridx4", "gridx5", "gridx6",
    # dispersion
    "d3", "d3bj", "d3zero", "d4", "abc",
    # printing / misc
    "miniprint", "smallprint", "largeprint", "normalprint", "printbasis",
    "nopop", "bohrs", "angs", "xyzfile", "pal2", "pal4", "pal8", "pal16",
}


class OrcaParser:
    """Parse an ORCA input file (.inp) into a QChemJob."""

    def parse(self, text: str, job_id: str = "orca_job") -> QChemJob:
        """Parse ORCA input and return QChemJob instance."""
        text = self._strip_comments(text)
        
        # Parse components
        atoms, charge, multiplicity = self._parse_coords(text)
        method, basis = self._parse_simple_input(text)
        job_type = self._parse_job_type(text)
        solvent = self._parse_solvent(text)
        nprocs = self._parse_nprocs(text)
        memory_mb = self._parse_memory(text)
        
        return QChemJob(
            id=job_id,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            atoms=atoms,
            job_type=job_type,
            engine="pyscf",
            solvent=solvent,
            nprocs=nprocs,
            memory_mb=memory_mb
        )

    # ------------------------------------------------------------------
    def _strip_comments(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            # ORCA comments start with #
            line = re.sub(r'#.*$', '', line)
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _parse_coords(self, text: str) -> tuple[list[tuple[str, float, float, float]], int, int]:
        """Extract coordinates and charge/mult from * xyz or * xyzfile."""
        atoms = []
        charge, mult = 0, 1

        coord_pattern = re.compile(
            r'\*\s*xyz\s*(-?\d+)\s+(\d+)\s+(.*?)(?=\*)', 
            re.DOTALL | re.IGNORECASE
        )
        
        m = coord_pattern.search(text)
        if m:
            charge = int(m.group(1))
            mult = int(m.group(2))
            coord_block = m.group(3).strip()
            
            for line in coord_block.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    sym = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((sym, x, y, z))
                    except ValueError:
                        continue
        return atoms, charge, mult

    # ------------------------------------------------------------------
    def _parse_simple_input(self, text: str) -> tuple[str, str]:
        """Extract method and basis from simple input line (!).

        ORCA route lines mix the method and basis with run-type keywords
        (SP, Opt, Freq) and accuracy/auxiliary modifiers (TightSCF, RIJCOSX,
        D3BJ, ...). Those are NOT methods, so they must be skipped — otherwise
        a trailing keyword like "SP" gets mistaken for the method.
        """
        method = None
        basis_name = "sto-3g"

        # Look for ! B3LYP def2-TZVP etc.
        simple_pattern = re.compile(r'^\s*!\s+(.*)', re.MULTILINE | re.IGNORECASE)
        for m in simple_pattern.finditer(text):
            # Stop at literal \n (escaped newline) in case the string wasn't fully decoded
            line_content = m.group(1).partition("\\n")[0]
            tokens = line_content.split()
            for tok in tokens:
                tok_lower = tok.lower()
                if any(x in tok_lower for x in ('def2', 'cc-p', 'aug-', 'pcseg', 'ano', 'sto', '6-31', '6-311')):
                    basis_name = basis(tok.strip())
                elif tok_lower in _ORCA_NON_METHOD_KEYWORDS:
                    continue
                elif method is None:
                    # First non-basis, non-keyword token is the method
                    method = tok.strip()

        return method or "hf", basis_name

    # ------------------------------------------------------------------
    def _parse_job_type(self, text: str) -> str:
        """Determine job type from simple input or blocks."""
        text_lower = text.lower()
        if 'freq' in text_lower or 'numfreq' in text_lower:
            return "freq"
        elif 'opt' in text_lower:
            return "opt"
        else:
            return "energy"

    # ------------------------------------------------------------------
    def _parse_solvent(self, text: str) -> Optional[str]:
        """Extract solvent model."""
        if re.search(r'cpcm|smd', text, re.IGNORECASE):
            return "pcm"
        return None

    # ------------------------------------------------------------------
    def _parse_memory(self, text: str) -> int:
        """Extract memory in MB."""
        # Look for %maxcore
        mem_pattern = re.compile(r'%maxcore\s+(\d+)', re.IGNORECASE)
        m = mem_pattern.search(text)
        if m:
            return int(m.group(1))
        return 4000

    # ------------------------------------------------------------------
    def _parse_nprocs(self, text: str) -> int:
        """Extract number of processors."""
        # Look for %pal nprocs
        nproc_pattern = re.compile(r'%pal\s+nprocs\s+(\d+)', re.IGNORECASE)
        m = nproc_pattern.search(text)
        if m:
            return int(m.group(1))
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Format auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(text: str, filename: str) -> str:
    """Guess whether a file is Psi4 or ORCA format."""
    fname = filename.lower()
    
    # Extension hints
    if fname.endswith('.dat'):
        return 'psi4'

    # Content clues
    if re.search(r'^\s*molecule\s*\w*\s*\{', text, re.MULTILINE | re.IGNORECASE):
        return 'psi4'
    if re.search(r'^\s*!', text, re.MULTILINE):
        return 'orca'
    if re.search(r'\*\s+xyz\s+-?\d+\s+\d+', text, re.IGNORECASE):
        return 'orca'
    if re.search(r'energy\s*\(|optimize\s*\(|frequencies\s*\(', text, re.IGNORECASE):
        return 'psi4'

    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse(text: str, fmt: str = "auto", source_name: str = "input", job_id: str = "job") -> QChemJob:
    """
    Parse a Psi4 or ORCA input string into a QChemJob specification.

    Parameters
    ----------
    text        : raw content of the Psi4 / ORCA input file
    fmt         : 'psi4', 'orca', or 'auto'
    source_name : original filename (used for auto-detection)
    job_id      : unique identifier for the QChemJob

    Returns
    -------
    QChemJob – normalized job specification
    """
    if fmt == "auto":
        fmt = detect_format(text, source_name)
        if fmt == "unknown":
            raise ValueError(
                "Cannot auto-detect format. "
                "Pass fmt='psi4' or fmt='orca' explicitly."
            )

    if fmt == "psi4":
        job = Psi4Parser().parse(text, job_id=job_id)
    elif fmt == "orca":
        job = OrcaParser().parse(text, job_id=job_id)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'psi4' or 'orca'.")

    return job
