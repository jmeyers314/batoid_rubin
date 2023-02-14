__version__ = "0.2.1"
tmp = __version__
if "rc" in tmp:  # pragma: no cover
    tmp = tmp[:tmp.find("rc")]
__version_info__ = tuple(map(int, tmp.split('.')))

from .builder import *
from .visualize import *
from .align_game import AlignGame

from pathlib import Path
datadir = Path(__file__).parent / "data"
