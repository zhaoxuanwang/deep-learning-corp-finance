"""src package — v2 is the active codebase; legacy shims for backwards compat.

Installs a meta-path finder that redirects `src.<pkg>` imports to
`src._legacy.<pkg>` for archived v1 packages. This avoids circular
imports because the redirect happens at the import machinery level,
before any __init__.py code runs.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

# Packages that were moved from src/<pkg> to src/_legacy/<pkg>.
_LEGACY_PACKAGES = frozenset({
    "economy", "ddp", "experimental", "networks",
    "pipeline", "trainers", "utils",
})


class _LegacyRedirectFinder(importlib.abc.MetaPathFinder):
    """Redirects src.<pkg>.* imports to src._legacy.<pkg>.*"""

    def find_spec(self, fullname, path, target=None):
        # Handle src._defaults -> src._legacy._defaults
        if fullname == "src._defaults":
            legacy_name = "src._legacy._defaults"
            return self._make_alias_spec(fullname, legacy_name)

        # Handle src.<legacy_pkg>.* -> src._legacy.<legacy_pkg>.*
        parts = fullname.split(".")
        if len(parts) >= 2 and parts[0] == "src" and parts[1] in _LEGACY_PACKAGES:
            legacy_name = "src._legacy." + ".".join(parts[1:])
            return self._make_alias_spec(fullname, legacy_name)

        return None

    def _make_alias_spec(self, alias_name, real_name):
        """Create a ModuleSpec that loads the real module under the alias name."""
        return importlib.machinery.ModuleSpec(
            alias_name,
            _LegacyRedirectLoader(real_name),
        )


class _LegacyRedirectLoader(importlib.abc.Loader):
    """Loader that imports the real _legacy module and caches it under the alias."""

    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        # Pre-set __path__ so that if the real module's __init__.py does
        # `from src.economy.submodule import ...`, Python sees this alias
        # as a package and can look up submodules via the finder.
        real_spec = importlib.util.find_spec(self.real_name)
        if real_spec and real_spec.submodule_search_locations:
            module.__path__ = list(real_spec.submodule_search_locations)

        real_mod = importlib.import_module(self.real_name)
        # Replace the alias with the real module in sys.modules
        sys.modules[module.__name__] = real_mod


# Install the finder at the beginning of sys.meta_path so it takes priority.
_finder = _LegacyRedirectFinder()
if _finder not in sys.meta_path:
    sys.meta_path.insert(0, _finder)
