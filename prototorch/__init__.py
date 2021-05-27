"""ProtoTorch package."""

import pkgutil

import pkg_resources

from . import components, datasets, functions, modules, utils
from .datasets import *

# Core Setup
__version__ = "0.4.5"

__all_core__ = [
    "datasets",
    "functions",
    "modules",
    "components",
    "utils",
]

# Plugin Loader
__path__ = pkgutil.extend_path(__path__, __name__)


def discover_plugins():
    return {
        entry_point.name: entry_point.load()
        for entry_point in pkg_resources.iter_entry_points(
            "prototorch.plugins")
    }


discovered_plugins = discover_plugins()
locals().update(discovered_plugins)

# Generate combines __version__ and __all__
version_plugins = "\n".join([
    "- " + name + ": v" + plugin.__version__
    for name, plugin in discovered_plugins.items()
])
if version_plugins != "":
    version_plugins = "\nPlugins: \n" + version_plugins

version = "core: v" + __version__ + version_plugins
__all__ = __all_core__ + list(discovered_plugins.keys())
