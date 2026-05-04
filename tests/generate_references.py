"""
Generate reference energies for the molecule benchmark dataset by running
PySCF locally on each (molecule, method, basis) combination.

Run once and commit the resulting `tests/data/references.json`. The agent
test then compares its `run_pyscf` output against these references with a
tight tolerance — if numbers match to ~µHartree, the agent really executed
PySCF; if they don't, it hallucinated.

Usage:
    python -m tests.generate_references
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .data.molecules import DATASET, Molecule


REFERENCES_PATH = Path(__file__).parent / "data" / "references.json"


def _run_one(mol: Molecule) -> dict:
    """Build a PySCF Mole and run the requested calculation; return summary dict."""
    from pyscf import gto, scf, dft, mp, cc

    atom_str = "\n".join(
        f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, x, y, z in mol["atoms"]
    )
    spin = mol["multiplicity"] - 1

    mole = gto.M(
        atom=atom_str,
        basis=mol["basis"],
        charge=mol["charge"],
        spin=spin,
        unit="Angstrom",
        verbose=0,
    )

    method = mol["method"].lower()
    is_open_shell = spin > 0
    t0 = time.time()

    if method == "hf":
        mf = (scf.UHF if is_open_shell else scf.RHF)(mole).run()
        energy = mf.e_tot
        converged = bool(mf.converged)
    elif method == "b3lyp":
        mf = (dft.UKS if is_open_shell else dft.RKS)(mole)
        mf.xc = "b3lyp"
        mf = mf.run()
        energy = mf.e_tot
        converged = bool(mf.converged)
    elif method == "mp2":
        mf = (scf.UHF if is_open_shell else scf.RHF)(mole).run()
        mp2 = mp.MP2(mf).run()
        energy = mf.e_tot + mp2.e_corr
        converged = bool(mf.converged)
    elif method == "ccsd":
        mf = (scf.UHF if is_open_shell else scf.RHF)(mole).run()
        ccsd = cc.CCSD(mf).run()
        energy = mf.e_tot + ccsd.e_corr
        converged = bool(mf.converged and ccsd.converged)
    else:
        raise ValueError(f"Unsupported method: {method}")

    wall = time.time() - t0

    dipole = None
    try:
        dip_xyz = mf.dip_moment(unit="Debye", verbose=0)
        dipole = [float(d) for d in dip_xyz]
    except Exception:
        pass

    return {
        "energy_ha": float(energy),
        "converged": converged,
        "dipole_debye": dipole,
        "wall_seconds": round(wall, 3),
        "n_atoms": len(mol["atoms"]),
    }


def main() -> int:
    refs: dict[str, dict] = {}
    for mol in DATASET:
        key = f"{mol['name']}__{mol['method']}__{mol['basis']}__{mol['job_type']}"
        print(f"[{key}] running ...", flush=True)
        try:
            result = _run_one(mol)
        except Exception as exc:
            print(f"[{key}] FAILED: {exc}", flush=True)
            refs[key] = {"error": str(exc)}
            continue
        print(
            f"[{key}] E = {result['energy_ha']:.8f} Ha "
            f"({result['wall_seconds']}s, converged={result['converged']})",
            flush=True,
        )
        refs[key] = result

    REFERENCES_PATH.write_text(json.dumps(refs, indent=2))
    print(f"\nWrote {len(refs)} reference entries to {REFERENCES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
