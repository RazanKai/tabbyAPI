"""Pytest bootstrap for the orchestration tests.

Upstream's test directory is not a package (no ``__init__.py``), so this conftest
exists only to make ``orch_helpers`` importable by name alongside the other test
modules and to expose the shared fixtures.
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent.parent
for path in (str(_HERE), str(_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from orch_helpers import (  # noqa: E402,F401
    FakeClock,
    FakeDeps,
    FakeTelemetry,
    MIB,
    calibrated_config,
    clock,
    deps,
    make_snapshot,
)
