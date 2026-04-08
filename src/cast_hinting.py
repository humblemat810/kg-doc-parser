from __future__ import annotations
from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")

def cached(memory: Memory, fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn))

memory = Memory(".cache", verbose=0)

def compute(a: int, b: float, *, mode: str = "x") -> tuple[int, float]:
    return (a, b)

compute_cached = cached(memory, compute)

# Type checker should now know:
# (a: int, b: float, *, mode: str = ...) -> tuple[int, float]