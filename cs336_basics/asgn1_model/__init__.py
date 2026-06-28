import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_base")
except importlib.metadata.PackageNotFoundError:
    pass
