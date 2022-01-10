"""ProtoTorch package"""

import pkgutil

import pkg_resources

from . import datasets  # noqa: F401
from . import nn  # noqa: F401
from . import utils  # noqa: F401
from .core import competitions  # noqa: F401
from .core import components  # noqa: F401
from .core import distances  # noqa: F401
from .core import initializers  # noqa: F401
from .core import losses  # noqa: F401
from .core import pooling  # noqa: F401
from .core import similarities  # noqa: F401
from .core import transforms  # noqa: F401

# Core Setup
__version__ = "0.7.1"

__all_core__ = [
    "competitions",
    "components",
    "core",
    "datasets",
    "distances",
    "initializers",
    "losses",
    "nn",
    "pooling",
    "similarities",
    "transforms",
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
