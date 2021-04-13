"""ProtoTorch package."""

# #############################################
# Core Setup
# #############################################
__version_core__ = "0.2.0-dev0"

from prototorch import datasets, functions, modules

__all_core__ = [
    "datasets",
    "functions",
    "modules",
]

# #############################################
# Plugin Loader
# #############################################
import pkg_resources


def discover_plugins():
    return {
        entry_point.name: entry_point.load()
        for entry_point in pkg_resources.iter_entry_points("prototorch.plugins")
    }


discovered_plugins = discover_plugins()
locals().update(discovered_plugins)

# Generate combines __version__ and __all__
__version_plugins__ = "\n".join(
    [
        "- " + name + ": v" + plugin.__version__
        for name, plugin in discovered_plugins.items()
    ]
)
if __version_plugins__ != "":
    __version_plugins__ = "\nPlugins: \n" + __version_plugins__

__version__ = "core: v" + __version_core__ + __version_plugins__
__all__ = __all_core__ + list(discovered_plugins.keys())