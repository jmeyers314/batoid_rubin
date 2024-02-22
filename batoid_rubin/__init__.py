from ._version import __version__, __version_tuple__

from .builder import *

from pathlib import Path
datadir = Path(__file__).parent / "data"
