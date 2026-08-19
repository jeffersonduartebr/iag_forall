# Objective: Package initialization and import surface for app.
"""Package initialization and import surface for app.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""

from __future__ import annotations

from pathlib import Path

_package_dir = Path(__file__).resolve().parent
_nested_package_dir = _package_dir / "app"

# The container image uses `/app` as the working directory, while local test
# runs commonly put `/repo/app` on `PYTHONPATH`. Extending `__path__` lets
# `import app.main` resolve in both layouts without forcing callers to know
# whether the runtime package root is `/app` or `/app/app`.
if _nested_package_dir.is_dir():
    __path__.append(str(_nested_package_dir))

