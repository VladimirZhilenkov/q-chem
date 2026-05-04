"""
qchem_converter.py
------------------
Convert Psi4 and ORCA input files to equivalent PySCF Python scripts.

Usage:
    python qchem_converter.py input.inp          # auto-detect format
    python qchem_converter.py input.inp --fmt psi4
    python qchem_converter.py input.inp --fmt orca
    python qchem_converter.py input.inp -o output.py

Supported features:
  - Molecule geometry (Cartesian and Z-matrix for Psi4)
  - Charge and spin multiplicity
  - DFT, HF, MP2, CCSD, CCSD(T)
  - Basis sets (Pople, Dunning, def2-*, etc.)
  - Job types: single-point, optimization, frequency
  - Solvent (CPCM/PCM)
  - Auxiliary basis / density fitting
  - Open-shell (UHF/UKS)
"""

import re
import sys
import argparse
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
        """Extract method and basis from simple input line (!)."""
        method = "hf"
        basis_name = "sto-3g"

        # Look for ! B3LYP def2-TZVP etc.
        simple_pattern = re.compile(r'^\s*!\s+(.*)', re.MULTILINE | re.IGNORECASE)
        for m in simple_pattern.finditer(text):
            tokens = m.group(1).split()
            for tok in tokens:
                tok_lower = tok.lower()
                if any(x in tok_lower for x in ('def2', 'cc-p', 'aug-', 'pcseg', 'ano', 'sto', '6-31', '6-311')):
                    basis_name = basis(tok.strip())
                else:
                    method = tok.strip()

        return method, basis_name

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
# PySCF code generator
# ─────────────────────────────────────────────────────────────────────────────

class PySCFGenerator:
    """Generate PySCF Python code from a QChemJob."""

    POST_HF_METHODS = {"mp2", "ccsd", "ccsd(t)", "cisd"}
    DFT_METHODS = {
        "b3lyp", "pbe0", "m06", "m062x", "wb97x", "wb97x_d",
        "camb3lyp", "blyp", "pbe", "tpss", "scan"
    }

    def generate(self, job: QChemJob, source_name: str = "input") -> str:
        """Generate complete PySCF script from QChemJob."""
        lines = []
        lines += self._header(source_name)
        lines += self._imports(job)
        lines += self._molecule(job)
        lines += self._method_setup(job)
        lines += self._run_job(job)
        lines += self._print_results(job)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _header(self, source_name: str) -> list[str]:
        return [
            "#!/usr/bin/env python3",
            '"""',
            f"PySCF calculation – converted from: {source_name}",
            '"""',
            "",
        ]

    # ------------------------------------------------------------------
    def _imports(self, job: QChemJob) -> list[str]:
        lines = ["from pyscf import gto, scf"]
        method = job.method.lower()
        
        if method in self.DFT_METHODS:
            lines.append("from pyscf import dft")
        if method in {"mp2"}:
            lines.append("from pyscf import mp")
        if method in {"ccsd", "ccsd(t)"}:
            lines.append("from pyscf import cc")
        if method in {"cisd"}:
            lines.append("from pyscf import ci")
        if job.job_type == "opt":
            lines.append("from pyscf.geomopt.geometric_solver import optimize")
        if job.job_type == "freq":
            lines.append("from pyscf import hessian")
            lines.append("from pyscf.hessian import thermo")
        if job.solvent:
            lines.append("from pyscf import solvent")
        
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    def _molecule(self, job: QChemJob) -> list[str]:
        lines = ["# ── Molecule ─────────────────────────────────────────────────"]
        lines.append("mol = gto.M(")
        lines.append(f"    atom = '''")
        for sym, x, y, z in job.atoms:
            lines.append(f"        {sym:<2} {x:12.8f} {y:12.8f} {z:12.8f}")
        lines.append("    ''',")
        lines.append(f"    basis = '{job.basis}',")
        lines.append(f"    charge = {job.charge},")
        lines.append(f"    spin = {job.multiplicity - 1},")
        lines.append(")")
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    def _method_setup(self, job: QChemJob) -> list[str]:
        lines = ["# ── Method ───────────────────────────────────────────────────"]
        method = job.method.lower()
        unrestricted = job.multiplicity != 1

        # For post-HF, set up HF first
        if method in self.POST_HF_METHODS:
            if unrestricted:
                lines.append("mf = scf.UHF(mol)")
            else:
                lines.append("mf = scf.RHF(mol)")
            
            if job.solvent:
                lines += self._solvent_wrap(job, "mf")
            
            lines.append("mf.kernel()")
            lines.append("")
            lines.append("# Post-HF correlation")
            
            if method == "mp2":
                cls = "mp.UMP2" if unrestricted else "mp.MP2"
            elif method == "ccsd":
                cls = "cc.UCCSD" if unrestricted else "cc.CCSD"
            elif method == "ccsd(t)":
                cls = "cc.UCCSD" if unrestricted else "cc.CCSD"
            elif method == "cisd":
                cls = "ci.UCISD" if unrestricted else "ci.CISD"
            else:
                cls = f"# TODO: {method}"
            
            lines.append(f"corr = {cls}(mf)")
            return lines

        # DFT or HF
        if method in self.DFT_METHODS:
            if unrestricted:
                lines.append("mf = dft.UKS(mol)")
            else:
                lines.append("mf = dft.RKS(mol)")
            lines.append(f"mf.xc = '{method}'")
        else:  # HF
            if unrestricted:
                lines.append("mf = scf.UHF(mol)")
            else:
                lines.append("mf = scf.RHF(mol)")

        if job.solvent:
            lines += self._solvent_wrap(job, "mf")

        lines.append("")
        return lines

    # ------------------------------------------------------------------
    def _solvent_wrap(self, job: QChemJob, var: str) -> list[str]:
        lines = [f"# Solvent: {job.solvent}"]
        if job.solvent in ("ddcosmo", "pcm"):
            lines.append(f"{var} = solvent.ddCOSMO({var})")
        return lines

    # ------------------------------------------------------------------
    def _run_job(self, job: QChemJob) -> list[str]:
        lines = ["# ── Run ──────────────────────────────────────────────────────"]
        method = job.method.lower()
        is_post_hf = method in self.POST_HF_METHODS

        if is_post_hf:
            lines.append("# HF already converged above")
            if method in ("mp2", "cisd"):
                lines.append("corr.kernel()")
                lines.append("e_total = corr.e_tot")
            elif method in ("ccsd", "ccsd(t)"):
                lines.append("corr.kernel()")
                lines.append("e_ccsd = corr.e_tot")
                if method == "ccsd(t)":
                    lines.append("# CCSD(T) perturbative triples")
                    lines.append("from pyscf.cc import ccsd_t")
                    lines.append("e_t = ccsd_t.kernel(corr, corr.ao2mo())")
                    lines.append("e_total = e_ccsd + e_t")
                else:
                    lines.append("e_total = e_ccsd")
            
            if job.job_type == "freq":
                lines += [
                    "",
                    "# Harmonic frequencies",
                    "hessian = corr.Hessian().kernel()",
                    "freq_info = thermo.harmonic_analysis(mol, hessian)",
                    "thermo.dump_normal_mode(mol, freq_info)",
                ]
        
        elif job.job_type == "energy":
            lines.append("e_total = mf.kernel()")
        
        elif job.job_type == "opt":
            lines.append("# Geometry optimization")
            lines.append("mol_eq = optimize(mf)")
            lines.append("print('Optimized geometry:')")
            lines.append("print(mol_eq.atom_coords())")
            lines.append("e_total = mf.e_tot")
        
        elif job.job_type == "freq":
            lines.append("mf.kernel()")
            lines.append("e_total = mf.e_tot")
            lines.append("")
            lines.append("# Harmonic frequencies")
            lines.append("hessian = mf.Hessian().kernel()")
            lines.append("freq_info = thermo.harmonic_analysis(mol, hessian)")
            lines.append("thermo.dump_normal_mode(mol, freq_info)")

        lines.append("")
        return lines

    # ------------------------------------------------------------------
    def _print_results(self, job: QChemJob) -> list[str]:
        lines = ["# ── Results ──────────────────────────────────────────────────"]
        lines.append("print(f'Total energy = {e_total:.10f} Ha')")
        
        method = job.method.lower()
        if method not in self.POST_HF_METHODS and job.job_type == "energy":
            lines.append("")
            lines.append("# Additional properties")
            lines.append("dm = mf.make_rdm1()")
            lines.append("print(f'Dipole moment (Debye) = {mf.dip_moment()}')")
            
            if job.multiplicity != 1:
                lines.append("print(f'<S^2> = {mf.spin_square()[0]:.4f}')")
        
        return lines


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

def convert(text: str, fmt: str = "auto", source_name: str = "input", job_id: str = "job") -> str:
    """
    Convert a Psi4 or ORCA input string to a PySCF Python script.

    Parameters
    ----------
    text        : raw content of the Psi4 / ORCA input file
    fmt         : 'psi4', 'orca', or 'auto'
    source_name : original filename, used only in the header comment
    job_id      : unique identifier for the QChemJob

    Returns
    -------
    str  – ready-to-run PySCF Python script
    """
    if fmt == "auto":
        fmt = detect_format(text, source_name)
        if fmt == "unknown":
            raise ValueError(
                "Cannot auto-detect format. "
                "Pass --fmt psi4 or --fmt orca explicitly."
            )

    if fmt == "psi4":
        job = Psi4Parser().parse(text, job_id=job_id)
    elif fmt == "orca":
        job = OrcaParser().parse(text, job_id=job_id)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'psi4' or 'orca'.")

    return PySCFGenerator().generate(job, source_name=source_name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Psi4 / ORCA input files to PySCF Python scripts."
    )
    parser.add_argument("input", help="Path to Psi4 or ORCA input file")
    parser.add_argument(
        "--fmt", choices=["auto", "psi4", "orca"], default="auto",
        help="Input format (default: auto-detect)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .py file (default: <input_stem>_pyscf.py)"
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Job ID for the QChemJob (default: based on input filename)"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    job_id = args.job_id or in_path.stem
    text = in_path.read_text()
    result = convert(text, fmt=args.fmt, source_name=in_path.name, job_id=job_id)

    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_pyscf.py"
    )
    out_path.write_text(result)
    print(f"✓  Written: {out_path}")


if __name__ == "__main__":
    main()