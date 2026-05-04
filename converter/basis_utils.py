import basis_set_exchange as bse 
from functools import lru_cache

@lru_cache(maxsize=128)

def check_existence(basis_name: str) -> bool:
    """Checking basis existance in BSE"""
    try:
        bse.get_basis(basis_name, fmt='json')
        return True
    except Exception:
        return False
