# Objective: Functional dependency algorithms used to compute gold answers for database design items.
"""Relational design algorithms, used to *compute* the gold answer.

Attribute closure, candidate keys and the highest normal form of a schema are
all decidable, so the gold answer of a normalization item never has to be
written by hand — which is precisely why these items can be graded by program
while the legacy catalog's "explain normalization" items could not.
"""

from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, List, Sequence, Tuple

FD = Tuple[FrozenSet[str], FrozenSet[str]]


def closure(attributes: FrozenSet[str], fds: Sequence[FD]) -> FrozenSet[str]:
    """Attribute closure under a set of functional dependencies."""
    result = set(attributes)
    changed = True
    while changed:
        changed = False
        for left, right in fds:
            if left <= result and not right <= result:
                result |= right
                changed = True
    return frozenset(result)


def is_superkey(attributes: FrozenSet[str], all_attrs: FrozenSet[str], fds: Sequence[FD]) -> bool:
    """True when the attributes functionally determine the whole relation."""
    return closure(attributes, fds) == all_attrs


def candidate_keys(all_attrs: FrozenSet[str], fds: Sequence[FD]) -> List[FrozenSet[str]]:
    """Every minimal superkey, by exhaustive search over subsets.

    Exhaustive is fine here: the generated relations have at most six
    attributes, and being exhaustive is what guarantees the gold answer.
    """
    found: List[FrozenSet[str]] = []
    ordered = sorted(all_attrs)
    for size in range(1, len(ordered) + 1):
        for combo in combinations(ordered, size):
            candidate = frozenset(combo)
            if any(key <= candidate for key in found):
                continue  # not minimal
            if is_superkey(candidate, all_attrs, fds):
                found.append(candidate)
    return found


def prime_attributes(all_attrs: FrozenSet[str], fds: Sequence[FD]) -> FrozenSet[str]:
    """Attributes taking part in at least one candidate key."""
    return frozenset().union(*candidate_keys(all_attrs, fds)) if all_attrs else frozenset()


def highest_normal_form(all_attrs: FrozenSet[str], fds: Sequence[FD]) -> str:
    """Highest normal form satisfied by the schema: ``BCNF``, ``3NF``, ``2NF`` or ``1NF``.

    Assumes the relation is already in 1NF (atomic attributes), which is the
    case for every generated schema.
    """
    keys = candidate_keys(all_attrs, fds)
    prime = frozenset().union(*keys) if keys else frozenset()
    non_trivial = [(left, right - left) for left, right in fds if right - left]

    if all(is_superkey(left, all_attrs, fds) for left, _ in non_trivial):
        return "BCNF"

    if all(
        is_superkey(left, all_attrs, fds) or (right <= prime)
        for left, right in non_trivial
    ):
        return "3NF"

    # 2NF fails only on a partial dependency: a non-prime attribute determined
    # by a proper subset of some candidate key.
    for left, right in non_trivial:
        if not right - prime:
            continue
        for key in keys:
            if left < key:
                return "1NF"
    return "2NF"


def is_lossless(
    all_attrs: FrozenSet[str], left: FrozenSet[str], right: FrozenSet[str], fds: Sequence[FD]
) -> bool:
    """Lossless-join test for a binary decomposition."""
    if left | right != all_attrs:
        return False
    shared = left & right
    if not shared:
        return False
    determined = closure(shared, fds)
    return left <= determined or right <= determined


def format_fds(fds: Sequence[FD]) -> str:
    """Render dependencies the way a textbook statement writes them."""
    return "; ".join(f"{''.join(sorted(left))} -> {''.join(sorted(right))}" for left, right in fds)
