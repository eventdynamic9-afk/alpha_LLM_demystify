"""Typed expression tree shared by every parser, serializer, executor and verifier (§6.1).

A formula is an immutable tree of :class:`Node` objects.  Leaves are fields (``op == "$"``) and
constants (``op == "#"``); internal nodes reference an operator in :mod:`dsl.operators`, carry
expression children and integer/float parameters (windows, lags, quantile levels, exponents).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple["Node", ...] = ()
    params: tuple = ()

    # ------------------------------------------------------------------ leaves
    @property
    def is_field(self) -> bool:
        return self.op == "$"

    @property
    def is_const(self) -> bool:
        return self.op == "#"

    @property
    def is_leaf(self) -> bool:
        return self.op in ("$", "#")

    @property
    def name(self) -> str:
        if not self.is_field:
            raise AttributeError("only field nodes have a name")
        return self.params[0]

    @property
    def value(self) -> float:
        if not self.is_const:
            raise AttributeError("only constant nodes have a value")
        return self.params[0]

    def __str__(self) -> str:  # pragma: no cover - convenience
        from .serialize import to_qlib

        return to_qlib(self)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"Node<{self}>"


def F(name: str) -> Node:
    """Field leaf."""
    return Node("$", (), (name,))


def C(value: float) -> Node:
    """Constant leaf (stored as float; -0.0 normalized to 0.0)."""
    v = float(value)
    if v == 0.0:
        v = 0.0
    return Node("#", (), (v,))


def op(name: str, *children: Node, params: tuple = ()) -> Node:
    return Node(name, tuple(children), tuple(params))


def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal."""
    yield node
    for ch in node.children:
        yield from walk(ch)


def transform(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Bottom-up rewrite: children first, then ``fn`` on the rebuilt node."""
    if node.children:
        node = Node(node.op, tuple(transform(c, fn) for c in node.children), node.params)
    return fn(node)


def replace_at(node: Node, path: tuple[int, ...], new: Node) -> Node:
    """Return a copy of ``node`` with the subtree at ``path`` (child indices) replaced."""
    if not path:
        return new
    i, rest = path[0], path[1:]
    kids = list(node.children)
    kids[i] = replace_at(kids[i], rest, new)
    return Node(node.op, tuple(kids), node.params)


def paths(node: Node, prefix: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], Node]]:
    yield prefix, node
    for i, ch in enumerate(node.children):
        yield from paths(ch, prefix + (i,))


def subtree(node: Node, path: tuple[int, ...]) -> Node:
    for i in path:
        node = node.children[i]
    return node


def fields_of(node: Node) -> set[str]:
    return {n.name for n in walk(node) if n.is_field}


def size(node: Node) -> int:
    return sum(1 for _ in walk(node))


def depth(node: Node) -> int:
    if not node.children:
        return 1
    return 1 + max(depth(c) for c in node.children)
