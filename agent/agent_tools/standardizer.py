from langchain_core.tools import tool
from loguru import logger

# НЕ импортируем QChemJob здесь
# from schemas import QChemJob  # УБРАТЬ!

@tool(description="Convert raw quantum chemistry input (ORCA or Psi4) into a standardized QChemJob object.")
def standardize_chem_input(raw_format: str, input_format: str = "auto"):
    """Convert raw quantum chemistry input (ORCA or Psi4) into a standardized QChemJob object."""
    # Импортируем ВНУТРИ функции
    from converter.qchem_converter import OrcaParser, Psi4Parser
    from schemas import QChemJob
    
    try:
        if input_format == "auto":
            if "!" in raw_format or raw_format.strip().startswith("!"):
                input_format = "orca"
            elif "molecule" in raw_format:
                input_format = "psi4"
            else:
                raise ValueError("Cannot detect format")
            
        if input_format == "orca":
            parser = OrcaParser()
        elif input_format == "psi4":
            parser = Psi4Parser()
        else:
            raise ValueError(f"Unsupported format: {input_format}")
        
        job = parser.parse(raw_format)
        logger.info(f"Successfully parsed {input_format} input")
        return job
    
    except Exception as e:
        logger.error(f"Standardization failed: {e}")
        raise

if __name__ == "__main__":
    test_input = """
    ! B3LYP def2-TZVP
    * xyz 0 1
    H 0 0 0
    H 0 0 0.74
    *
    """

    res = standardize_chem_input.invoke({
        "raw_format": test_input,
        "input_format": "auto"
    })

    print(res)
