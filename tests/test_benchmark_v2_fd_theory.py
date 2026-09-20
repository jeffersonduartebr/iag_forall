# Objective: Tests for the functional dependency algorithms that compute gold answers for database items.
"""The normalization gold answers are only as trustworthy as these algorithms.

Every case below is a textbook example with a known answer, so a regression in
``closure``/``candidate_keys``/``highest_normal_form`` shows up here rather than
as a silently wrong gold value in the corpus.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from scripts.benchmark_v2.fd_theory import (  # noqa: E402
    candidate_keys,
    closure,
    highest_normal_form,
    is_lossless,
    is_superkey,
)


def fd(left, right):
    return frozenset(left), frozenset(right)


def test_closure_reaches_fixed_point_through_a_chain():
    fds = [fd("A", "B"), fd("B", "C"), fd("C", "D")]
    assert closure(frozenset("A"), fds) == frozenset("ABCD")
    assert closure(frozenset("C"), fds) == frozenset("CD")


def test_closure_of_an_attribute_with_no_dependency_is_itself():
    assert closure(frozenset("X"), [fd("A", "B")]) == frozenset("X")


def test_closure_requires_the_whole_left_hand_side():
    fds = [fd("AB", "C")]
    assert closure(frozenset("A"), fds) == frozenset("A")
    assert closure(frozenset("AB"), fds) == frozenset("ABC")


def test_superkey_detection():
    fds = [fd("A", "BC")]
    assert is_superkey(frozenset("A"), frozenset("ABC"), fds)
    assert not is_superkey(frozenset("B"), frozenset("ABC"), fds)


def test_candidate_keys_are_minimal():
    # R(A,B,C) with A -> BC: A is the only candidate key, and AB is not minimal.
    keys = candidate_keys(frozenset("ABC"), [fd("A", "BC")])
    assert keys == [frozenset("A")]


def test_candidate_keys_finds_every_alternative():
    # A -> B, B -> C, C -> A makes each single attribute a key.
    keys = candidate_keys(frozenset("ABC"), [fd("A", "B"), fd("B", "C"), fd("C", "A")])
    assert sorted(keys, key=sorted) == [frozenset("A"), frozenset("B"), frozenset("C")]


def test_candidate_key_can_be_composite():
    keys = candidate_keys(frozenset("ABC"), [fd("AB", "C")])
    assert keys == [frozenset("AB")]


@pytest.mark.parametrize(
    "attrs,fds,expected",
    [
        # Everything determined by the single key: BCNF.
        ("ABC", [fd("A", "BC")], "BCNF"),
        # Transitive dependency B -> C with key A: 2NF but not 3NF.
        ("ABC", [fd("A", "B"), fd("B", "C")], "2NF"),
        # Partial dependency A -> C on the composite key AB: only 1NF.
        ("ABC", [fd("AB", "C"), fd("A", "C")], "1NF"),
        # Determinant C is not a superkey but D is prime: 3NF, not BCNF.
        ("ABCD", [fd("AB", "CD"), fd("C", "A")], "3NF"),
        # No dependency at all: the whole relation is the key.
        ("ABC", [], "BCNF"),
    ],
)
def test_highest_normal_form(attrs, fds, expected):
    assert highest_normal_form(frozenset(attrs), fds) == expected


def test_lossless_when_the_shared_attribute_determines_one_side():
    fds = [fd("A", "B"), fd("A", "C")]
    assert is_lossless(frozenset("ABC"), frozenset("AB"), frozenset("AC"), fds)


def test_lossy_when_the_shared_attribute_determines_neither_side():
    fds = [fd("A", "B"), fd("C", "B")]
    assert not is_lossless(frozenset("ABC"), frozenset("AB"), frozenset("BC"), fds)


def test_decomposition_that_loses_an_attribute_is_never_lossless():
    fds = [fd("A", "B")]
    assert not is_lossless(frozenset("ABC"), frozenset("AB"), frozenset("AB"), fds)


def test_decomposition_without_shared_attributes_is_never_lossless():
    fds = [fd("A", "B")]
    assert not is_lossless(frozenset("AB"), frozenset("A"), frozenset("B"), fds)
