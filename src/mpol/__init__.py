zenodo_record = 10064221

import importlib.metadata

try:
    __version__ = importlib.metadata.version("MPoL")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"