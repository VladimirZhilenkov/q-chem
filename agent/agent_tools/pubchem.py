"""PubChem REST API tool — fetch molecular 3D structure and properties."""

import json
import requests
from langchain_core.tools import tool
from loguru import logger

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# fmt: off
_ELEMENT_SYMBOLS = {
    1:"H",  2:"He", 3:"Li", 4:"Be", 5:"B",  6:"C",  7:"N",  8:"O",  9:"F",  10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P", 16:"S", 17:"Cl",18:"Ar",
    19:"K", 20:"Ca",21:"Sc",22:"Ti",23:"V", 24:"Cr",25:"Mn",26:"Fe",
    27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",
    35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y", 40:"Zr",41:"Nb",42:"Mo",
    43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",
    51:"Sb",52:"Te",53:"I", 54:"Xe",55:"Cs",56:"Ba",57:"La",
    72:"Hf",73:"Ta",74:"W", 75:"Re",76:"Os",77:"Ir",78:"Pt",
    79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",92:"U",
}
# fmt: on


def _sym(atomic_number: int) -> str:
    return _ELEMENT_SYMBOLS.get(atomic_number, f"X{atomic_number}")


def _get_cid(name: str) -> int:
    """Resolve a molecule name to its PubChem CID."""
    resp = requests.get(
        f"{PUBCHEM_BASE}/compound/name/{name}/cids/JSON",
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["IdentifierList"]["CID"][0]


def _fetch_record(cid: int, record_type: str = "3d") -> dict | None:
    """Fetch a PubChem compound record; returns None on 404."""
    resp = requests.get(
        f"{PUBCHEM_BASE}/compound/cid/{cid}/record/JSON",
        params={"record_type": record_type},
        timeout=15,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _fetch_properties(cid: int) -> dict:
    """Fetch scalar properties (formula, weight, IUPAC name) from PubChem."""
    props = "MolecularFormula,MolecularWeight,IUPACName"
    resp = requests.get(
        f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON",
        timeout=15,
    )
    if resp.status_code != 200:
        return {}
    rows = resp.json().get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else {}


def _parse_record(data: dict) -> dict:
    """Extract atoms, 3D coordinates, charge, and named properties."""
    cmpd = data["PC_Compounds"][0]

    symbols = [_sym(z) for z in cmpd["atoms"]["element"]]

    # Grab the first conformer that has a z-array (→ 3D)
    coords = None
    for coord_block in cmpd.get("coords", []):
        for conf in coord_block.get("conformers", []):
            if "z" in conf:
                coords = list(zip(symbols, conf["x"], conf["y"], conf["z"]))
                break
        if coords:
            break

    charge = cmpd.get("charge", 0)

    iupac = formula = weight = ""
    for prop in cmpd.get("props", []):
        urn = prop.get("urn", {})
        val = prop.get("value", {})
        label, name_ = urn.get("label", ""), urn.get("name", "")
        if label == "IUPAC Name" and name_ == "Preferred":
            iupac = val.get("sval", "")
        elif label == "Molecular Formula":
            formula = val.get("sval", "")
        elif label == "Molecular Weight":
            weight = str(val.get("fval", val.get("sval", "")))

    return {"symbols": symbols, "coords": coords, "charge": charge,
            "iupac": iupac, "formula": formula, "weight": weight}


def _to_orca(mol: dict, method: str, basis: str, multiplicity: int, job_type: str) -> str:
    """Render an ORCA-format input string from parsed molecule data."""
    extra = {"opt": "Opt", "freq": "Freq"}.get(job_type, "")
    header = f"! {method} {basis} {extra}".rstrip()
    body = "\n".join(
        f"  {s}  {x:.6f}  {y:.6f}  {z:.6f}"
        for s, x, y, z in mol["coords"]
    )
    return f"{header}\n\n* xyz {mol['charge']} {multiplicity}\n{body}\n*\n"


@tool
def get_molecule_from_pubchem(query: str) -> str:
    """Fetch a molecule's 3D structure from PubChem and return its properties plus a ready-to-use ORCA input block.

    Use this tool when the user refers to a molecule by name (common or IUPAC), asks to
    'look up' a molecule, or needs geometry before running a calculation.

    Pass ONLY the molecule name as a plain string, e.g.:
        aspirin
        water
        caffeine
    Optionally pass a JSON object to control the calculation:
        {"molecule_name": "aspirin", "method": "B3LYP", "basis": "def2-SVP", "job_type": "energy"}

    Returns:
        Molecule summary (name, formula, weight, charge, atom count) followed by an
        ORCA input block that can be passed directly to run_pyscf.
    """
    # Unwrap JSON if the LLM passed the full Action Input dict as a single string
    molecule_name = query.strip()
    method = "B3LYP"
    basis = "def2-SVP"
    job_type = "energy"

    if molecule_name.startswith("{"):
        try:
            parsed = json.loads(molecule_name)
            if isinstance(parsed, dict):
                molecule_name = parsed.get("molecule_name", molecule_name)
                method = parsed.get("method", method)
                basis = parsed.get("basis", basis)
                job_type = parsed.get("job_type", job_type)
        except json.JSONDecodeError:
            pass

    logger.info(f"PubChem lookup: '{molecule_name}'")
    try:
        cid = _get_cid(molecule_name)
        logger.info(f"Resolved '{molecule_name}' → CID {cid}")

        data = _fetch_record(cid, "3d") or _fetch_record(cid, "2d")
        if data is None:
            return f"Error: no PubChem record found for CID {cid}."

        mol = _parse_record(data)
        if not mol["coords"]:
            return (
                f"PubChem returned no 3D coordinates for '{molecule_name}' (CID {cid}). "
                "Please provide the geometry manually."
            )

        # Supplement with the dedicated properties endpoint (richer than the record JSON)
        props = _fetch_properties(cid)
        formula = props.get("MolecularFormula") or mol["formula"]
        weight = str(props.get("MolecularWeight") or mol["weight"])
        iupac = props.get("IUPACName") or mol["iupac"] or molecule_name

        orca_block = _to_orca(mol, method, basis, 1, job_type)

        return (
            f"Molecule        : {iupac}\n"
            f"PubChem CID     : {cid}\n"
            f"Formula         : {formula}\n"
            f"Molecular weight: {weight} g/mol\n"
            f"Charge          : {mol['charge']}\n"
            f"Multiplicity    : 1  (assumed singlet; adjust if the molecule is a radical)\n"
            f"Atoms           : {len(mol['coords'])}\n"
            f"\nORCA input (pass this string directly to run_pyscf):\n{orca_block}"
        )

    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        if code == 404:
            return f"Molecule '{molecule_name}' not found in PubChem. Check spelling or try the IUPAC name."
        return f"PubChem HTTP {code}: {exc}"
    except Exception as exc:
        logger.error(f"PubChem error for '{molecule_name}': {exc}")
        return f"Error fetching from PubChem: {exc}"
