"""Make the project root importable so tests can `import schemas`, etc."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
