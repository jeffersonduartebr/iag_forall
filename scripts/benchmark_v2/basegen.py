# Objective: Shared base-item specification and generator registry for the benchmark v2 builders.
"""What a generator produces, and how generators are registered.

A generator never writes prose around a fixed answer; it draws parameters and
*computes* the gold value. That is the whole difference from the legacy catalog,
whose questions were templates with no answer at all.

Each spec may carry a second phrasing, ``query_dense``: the same problem stated
tersely, with the intermediate values the canonical form hands over left
implicit. Densification has to come from the generator because only it knows
which givens can be withheld without making the item ambiguous — a mechanical
"shorten this text" pass would produce unanswerable questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BaseSpec:
    """One generated problem, before it becomes corpus items."""

    topic: str
    query: str
    answer_type: str
    grader: str
    gold: Any
    steps: int
    tolerance: Optional[float] = None
    query_dense: Optional[str] = None
    steps_dense: int = 0
    rubric: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)
    provenance: str = "generated"

    @property
    def has_dense(self) -> bool:
        return bool(self.query_dense) and self.steps_dense >= 3


Generator = Callable[[Any], BaseSpec]

#: discipline -> list of generators
REGISTRY: Dict[str, List[Generator]] = {}


def generator(discipline: str) -> Callable[[Generator], Generator]:
    """Register a generator under one discipline."""

    def decorate(func: Generator) -> Generator:
        REGISTRY.setdefault(discipline, []).append(func)
        return func

    return decorate


def generators_for(discipline: str) -> List[Generator]:
    """Every generator registered for a discipline, in registration order."""
    return list(REGISTRY.get(discipline, []))


def fmt(value: float, places: int = 2) -> str:
    """Format a number the way a Portuguese statement writes it (comma decimals)."""
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.{places}f}".replace(".", ",")


def plural(count: int, singular: str, many: str) -> str:
    """Tiny agreement helper so generated statements read naturally."""
    return f"{count} {singular if count == 1 else many}"
