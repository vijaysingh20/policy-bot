import importlib
import pkgutil

import pytest

import backend

MODULES = sorted(
    name for _, name, _ in pkgutil.walk_packages(backend.__path__, prefix="backend.")
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)