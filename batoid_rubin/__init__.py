__version__ = "0.1.0"
tmp = __version__
if "rc" in tmp:  # pragma: no cover
    tmp = tmp[:tmp.find("rc")]
__version_info__ = tuple(map(int, tmp.split('.')))

from .builder import *
from .visualize import *
from .align_game import AlignGame

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
